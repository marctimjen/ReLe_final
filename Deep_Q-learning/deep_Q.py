# This file is created to make a small environment to test the models in.

# What I want to test firstly is creating a small environment that the player can learn to collect gold in.

# Idea 1: follow a A* path finding algo - to get the fastest to the coins.
# Idea 2: Lets make a small gym enviroment setup to learn the robots simply to collect gold.

import numpy as np

import sys
sys.path.append('..')  # Add the parent directory to sys.path
sys.path.append('../..')  # Add the parent directory to sys.path

import time
import collections
import torch
import torch.optim as optim
from rele_pack.converters import action_to_hot_gpu as action_to_hot, hot_to_action_gpu as hot_to_action
from rele_pack.net import DQN
from rele_pack.model_dir import model_dir, model_saver, model_path_loader, model_loader
from rele_pack.chest_env import chest_env
from rele_pack.loss import calc_loss_double_dqn
import neptune
import os

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
# class ExperienceBuffer:  # to keep past actions
#     def __init__(self, capacity):
#         self.buffer = collections.deque(maxlen=capacity)  # set buffer and size of que
#
#     def __len__(self):
#         return len(self.buffer)  # return lenght
#
#     def append(self, experience):
#         self.buffer.append(experience)  # append the experience to the end of the que
#
#     def sample(self, batch_size):
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)  # select random batches...
#         states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
#
#         return (states, actions, rewards, dones, next_states)  # get control over the states retuned!


class ExperienceBuffer:
    def __init__(self, capacity, state_shape: int = 6, device = 'cuda:0'):
        self.device = device
        self.sample_index = 0
        self.capacity = capacity

        self.state = torch.zeros((capacity, state_shape), dtype=torch.float64, device=device)
        self.next_state = torch.zeros((capacity, state_shape), dtype=torch.float64, device=device)

        self.rewards = torch.zeros((capacity, 1), dtype=torch.float64, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

        self.size = 0

    def __len__(self):
        return self.size

    def append(self, experience):
        state, action, reward, done, new_state = experience

        self.state[self.sample_index] = state
        self.next_state[self.sample_index] = new_state
        self.rewards[self.sample_index] = reward
        self.actions[self.sample_index] = action
        self.dones[self.sample_index] = done

        # Update the index and size, and wrap around if necessary
        self.sample_index = (self.sample_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Select random indices for the batch
        indices = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=self.device)

        return (self.state[indices], self.actions[indices], self.rewards[indices], self.dones[indices],
                self.next_state[indices])

class Agent:
    def __init__(self, env, exp_buffer, device="cpu"):
        self.env = env
        self.exp_buffer = exp_buffer  # get buffer
        self.device = device
        self._reset()

    def _reset(self):
        """
        This functions resets the env and sets the reward to 0.
        """
        self.state = env.reset()
        self.total_reward = torch.tensor((0), dtype=torch.float32, device=self.device)

    @torch.no_grad()  # we dont wan't to use gradients in out target network
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:  # network action or random action
            action = env.sample()  # make random action
        else:
            # state_a = np.array([self.state], copy=False)  # get state
            # state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(self.state)[0]  # give state to "normal" network
            _, act_v = torch.max(q_vals_v, dim=0)
            action = hot_to_action(act_v, device)  # get the action that is best based on network

        new_state, reward, is_done, _ = self.env.step(action)  # do step in the environment
        self.total_reward += reward

        exp = Experience(self.state, action_to_hot(action, device).view(-1), reward.view(-1), is_done.view(-1), new_state)
        self.exp_buffer.append(exp)  # append information to buffer
        self.state = new_state  # get new state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward  # return the reward when game is done else None

if __name__ == '__main__':
    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="ReL/ReLe-final",
        api_token=token,
    )

    run["Algo"] = "Deep_Q-learning"

    params = {"number_of_actions": 10,
              "grid_size": 5,
              "gamma": 0.99,
              "batch_size": 32,
              "replay_size": 2000000,
              "lr": 1e-4,
              "sync_target_frames": 10000,
              "replay_start_size": 50000,
              "epsilon_decay_last_frame": 500000,
              "epsilon_start": 0.99,
              "epsilon_final": 0.000001,
              "amount_of_eval_rounds": 450,  # how many games to ace in a row.
              }

    run["parameters"] = params

    script_directory = os.path.abspath(os.path.dirname(__file__))
    save_path = model_dir(script_directory)

    run["save_path"] = save_path

    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = DQN()
    net.to(device)
    tgt_net = DQN()
    tgt_net.to(device)

    env = chest_env(number_of_actions=params["number_of_actions"], grid_size=params["grid_size"], normalize=True,
                    return_distance=True, use_tensor=True)
    obs = env.reset()

    buffer = ExperienceBuffer(capacity=params["replay_size"], state_shape=10, device=device)
    agent = Agent(env, buffer, device=device)

    optimizer = optim.Adam(net.parameters(), lr=params["lr"])  # set optimizer
    total_rewards = torch.tensor([0], dtype=torch.float32, device=device)
    frame_idx = torch.tensor((0), dtype=torch.int, device=device)
    ts_frame = torch.tensor((0), dtype=torch.int, device=device)
    game_nr = torch.tensor((0), dtype=torch.int, device=device)
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(params["epsilon_final"], params["epsilon_start"] - frame_idx / params["epsilon_decay_last_frame"])
        reward = agent.play_step(net, epsilon, device=device)

        #  Only when a game is over then we return the reward (we return None while the game is in progress).
        if reward is not None:
            game_nr += 1
            total_rewards = torch.concat((total_rewards, reward.view(-1)))  # append the reward of the whole game
            speed = (frame_idx - ts_frame) / (time.time() - ts)  # Get speed per frame
            ts_frame = frame_idx.clone()
            ts = time.time()
            m_reward = torch.mean(total_rewards[-100:])  # reward of last 100 rewards
            # print(f"{frame_idx}: done {game_nr} games, reward {m_reward}," \
            #         + f" eps {epsilon}, speed {speed} f/s")

            run["game_nr"].log(game_nr)
            run["reward"].log(m_reward)
            run["epsilon"].log(epsilon)

            if speed == torch.tensor(float("inf")):  # this is not logable:
                speed = torch.tensor(1000000)

            run["speed"].log(speed)

            if best_m_reward is None or best_m_reward < m_reward:
                model_saver(model=net, save_path=save_path, game_idx=game_nr.item(), reward=m_reward.item())

                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward} -> {m_reward}")
                best_m_reward = m_reward  # Set the best reward as the last 100

            if torch.sum(total_rewards[-params["amount_of_eval_rounds"]:]) >= params["amount_of_eval_rounds"] - 0.1:
                # how many games do we need to play to beat the game (we need to win 19).
                print(f"Solved in {frame_idx} frames!")
                model_saver(model=net, save_path=save_path, game_idx=game_nr.item(), reward=m_reward.item(), final=True)
                run.stop()
                break

        if len(buffer) < params["replay_start_size"]:  # we need to fill the buffer before we start to train
            continue

        if frame_idx % params["sync_target_frames"] == 0:  # For every SYNC_TARGET_FRAMES=1000 frames we update the tgt network
            tgt_net.load_state_dict(net.state_dict())  # Load the newest version of the network

        optimizer.zero_grad()
        batch = buffer.sample(params["batch_size"])  # sample BATCH_SIZE=32 samples from the buffer (randomly)
                                           # Sampling only takes 4x frames randomly from the game-play
        loss_t = calc_loss_double_dqn(batch, net, tgt_net, gamma=params["gamma"], double=True)
        loss_t.backward()
        optimizer.step()
