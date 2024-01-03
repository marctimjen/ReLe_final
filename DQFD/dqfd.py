import numpy as np
import sys
sys.path.append('..')  # Add the parent directory to sys.path
sys.path.append('../..')  # Add the parent directory to sys.path

import os
import time
import collections
import torch.optim as optim
from rele_pack.converters import action_to_hot_gpu as action_to_hot, hot_to_action_gpu as hot_to_action
from rele_pack.net import DQN
from rele_pack.model_dir import model_dir, model_saver, model_path_loader, model_loader
from rele_pack.chest_env import chest_env
from rele_pack.loss import calc_loss_DQfD
from rele_pack.expert_policy import expert_policy
from rele_pack.replay_buffer import DQFD_experience_buffer
import torch
import torch.nn.utils as nn_utils
import neptune

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


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
        self.exp_buffer.clear_at_end_of_episode()
        self.state = self.env.reset()
        self.total_reward = torch.tensor((0), dtype=torch.float32, device=self.device)

    @torch.no_grad()  # we dont wan't to use gradients in out target network
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:  # network action or random action
            action = self.env.sample()  # make random action
        else:
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


def dqfd_main(params: dict):
    start = time.time()
    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="ReL/ReLe-final",
        api_token=token,
    )

    run["Algo"] = "DQFD"
    run["parameters"] = params
    lambda_loss = params["lambda_loss"]
    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    script_directory = os.path.abspath(os.path.dirname(__file__))
    save_path = model_dir(script_directory)

    CLIP_GRAD = 1.0  # Clip the gradient to this value

    net = DQN()
    net.to(device)
    tgt_net = DQN()
    tgt_net.to(device)

    env = chest_env(number_of_actions=params["number_of_actions"], grid_size=params["grid_size"], normalize=True,
                    return_distance=True, use_tensor=True)
    obs = env.reset()

    buffer = DQFD_experience_buffer(capacity=params["replay_size"], amount_expert=params["expert_play"], state_shape=10,
                                    prob_alpha=0.6, n_step=3, device=device, beta_start=params["beta_start"],
                                    beta_frames=params["beta_frames"], gamma=params["gamma"])

    # Start with the expert data:
    expert_pol = expert_policy()

    for i in range(params["expert_play"]):  # Now make the experts play :D
        action = expert_pol.play_step(obs)
        action = torch.tensor(action).view(2, 1)
        new_state, reward, is_done, _ = env.step(action)
        exp = Experience(obs, action_to_hot(action, device).view(-1), reward.view(-1), is_done.view(-1), new_state)
        buffer.append(exp)  # append information to buffer
        obs = new_state  # get new state
        if is_done:
            buffer.clear_at_end_of_episode()
            obs = env.reset()
            expert_pol = expert_policy()

    agent = Agent(env, buffer, device=device)

    optimizer = optim.Adam(net.parameters(), lr=params["lr"])  # set optimizer
    total_rewards = torch.tensor([0], dtype=torch.float32, device=device)
    frame_idx = torch.tensor((0), dtype=torch.int, device=device)
    ts_frame = torch.tensor((0), dtype=torch.int, device=device)
    game_nr = torch.tensor((0), dtype=torch.int, device=device)
    ts = time.time()
    best_m_reward = None
    pre_train_over = False

    while True:
        frame_idx += 1

        if pre_train_over:
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
                run["game_nr"].log(game_nr)
                run["reward/mean_reward"].log(m_reward)
                run["reward/epsilon"].log(epsilon)
                run["reward/beta"].log(buffer.beta)

                if speed == torch.tensor(float("inf")):  # this is not logable:
                    speed = torch.tensor(1000000)

                run["reward/speed"].log(speed)
                # print(f"{frame_idx}: done {game_nr} games, reward {m_reward}," \
                #         + f" eps {epsilon}, speed {speed} f/s")

                if best_m_reward is None or best_m_reward < m_reward:
                    model_saver(model=net, save_path=save_path, game_idx=game_nr.item(), reward=m_reward.item())

                    if best_m_reward is not None:
                        print(f"Best reward updated {best_m_reward} -> {m_reward}")
                    best_m_reward = m_reward  # Set the best reward as the last 100

                if torch.sum(total_rewards[-params["amount_of_eval_rounds"]:]) >= params["amount_of_eval_rounds"] - 0.1:  # how many games do we need to play to beat the game (we need to win 19).
                    print("Solved in %d frames!" % frame_idx)
                    model_saver(model=net, save_path=save_path, game_idx=game_nr.item(), reward=m_reward.item(), final=True)
                    break

            if len(buffer) < params["replay_start_size"]:  # we need to fill the buffer before we start to train
                continue
        else:
            if frame_idx > params["pre_train_phase"]:
                pre_train_over = True

        if frame_idx % params["sync_target_frames"] == 0:  # For every SYNC_TARGET_FRAMES=1000 frames we update the tgt network
            tgt_net.load_state_dict(net.state_dict())  # Load the newest version of the network

        optimizer.zero_grad()
        buf_sample = buffer.sample(params["batch_size"])  # sample BATCH_SIZE=32 samples from the buffer (randomly)
                                                # Sampling only takes 4x frames randomly from the game-play
        loss_t, prio = calc_loss_DQfD(buf_sample, net, tgt_net, gamma=params["gamma"], double=True, run=run,
                                      nr_exp_samples=params["expert_play"], lambdas=lambda_loss, device=device)
        buffer.update_priorities(buf_sample[0], prio)  # update the priorities of the samples
        buffer.update_beta(frame_idx)  # update the beta values so that beta -> 1 during training
        loss_t.backward()
        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        optimizer.step()

    end = time.time()
    total_time = end - start
    run["time"] = total_time
    run["iter_no"] = frame_idx
    run_id = run["sys/id"].fetch()
    run.stop()
    return total_time, frame_idx, run_id


if __name__ == "__main__":
    lambda_loss = {"lambda_dq": 1, "lambda_n": 1, "lambda_je": 1, "lambda_l2": 0.0005}

    # params = {"number_of_actions": 256,
    #           "grid_size": 128,
    #           "gamma": 0.99,
    #           "batch_size": 32,
    #           "replay_size": 200000,
    #           "lr": 1e-4,
    #           "sync_target_frames": 10000,
    #           "replay_start_size": 50000,
    #           "epsilon_decay_last_frame": 500000,
    #           "epsilon_start": 0.99,
    #           "epsilon_final": 0.000001,
    #           "beta_frames": 10000000,
    #           "beta_start": 0.4,
    #           "expert_play": 50000,  # The amount of expert frames!
    #           "pre_train_phase": 10000,
    #           "lambda_loss": lambda_loss,  # the weights for the different loss-functions
    #           "amount_of_eval_rounds": 450,  # how many games to ace in a row.
    #           }

    params = {"number_of_actions": 5,
              "grid_size": 2,
              "gamma": 0.99,
              "batch_size": 32,
              "replay_size": 200000,
              "lr": 1e-4,
              "sync_target_frames": 20000,
              "replay_start_size": 50000,
              "epsilon_decay_last_frame": 10000,
              "epsilon_start": 0.99,
              "epsilon_final": 0.000001,
              "beta_frames": 10000,
              "beta_start": 0.4,
              "expert_play": 5000,  # The amount of expert frames!
              "pre_train_phase": 1000,
              "lambda_loss": lambda_loss,  # the weights for the different loss-functions
              "amount_of_eval_rounds": 450,  # how many games to ace in a row.
              }

    dqfd_main(params)
