import sys
sys.path.append('..')  # Add the parent directory to sys.path
sys.path.append('../..')  # Add the parent directory to sys.path
import gym
import collections
from rele_pack.chest_env import chest_env
import torch
import neptune
import os
import time

# 0: (0, 0) # Move up and left
# 1: (0, 1) # Move up
# 2: (1, 0) # Move left
# 3: (1, 1) # stand still
# 4: (0, 2) # Move up and right
# 5: (2, 0) # Move down and left
# 6: (1, 2) # Move right
# 7: (2, 1) # Move down
# 8: (2, 2) # Move down and right


"""
This implementation of the Q-learning algorithm estimate the Q-values using the algorithm:

Q_n+1(x, a) = (1 - alpha_n) * Q_n(x, a) + alpha_n * (r(x, a) + lambda * max_b(Q_n(y, b)))
"""

class Agent:
    def __init__(self, env, gamma, decay):
        self.env = env
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.q_values = collections.defaultdict(float)  # table with Q-values
        self.gamma = gamma

        it = self.alpha_decay(decay=decay)
        self.alpha = lambda: next(it)  # get the next alpha value in the sequence

    def alpha_decay(self, lr: float = 1.0, decay: float = 0.01) -> float:
        """
        This function is for the decay of the alpha value.

        :param lr: (float) parameter that determine the first alpha value.
        :param decay: (float) how fast should the alpha values decay?
        :yield (float): the next alpha value
        """
        i = 0
        while True:
            yield lr / (i * decay + 1)
            i += 1

    def select_action(self, state, epsi: float = 0.0):
        """
        Based on the state that we are in select the best action (the one with best value).
        """
        best_action, best_value = None, None

        if torch.rand(1) < epsi:  # at random times take a random action instead of the best action
            return self.env.sample()

        gen_all_actions = self.env.iter_all_actions()  # get all the actions

        for action in gen_all_actions:  # all possible actions
            action_value = self.q_values[(state, action)]  # take the action that maximize the Q-function
            if best_value is None or best_value < action_value:  # get the best action
                best_value = action_value
                best_action = action

        return best_action

    def q_learn(self, nr_episodes: int, epsi: float):
        """
        Here we iterate over all states and actions and calculate the max state value. With this we can fill the values
        table.

        :return: None
        """

        is_done = False
        self.alp = self.alpha()  # get the alpha value for this iteration
        for _ in range(nr_episodes):
            while not is_done:
                action = self.select_action(self.state, epsi=epsi)  # get the best action
                new_state, reward, is_done, _ = self.env.step(action)  # take the action in the env
                self.rewards[(self.state, action, new_state)] = reward  # insert in reward table
                self.transits[(self.state, action)][new_state] += 1  # increment with 1 in the transaction table

                q_now = self.q_values[(self.state, action)]
                best_action = self.select_action(new_state)  # take the best action for the next state
                q_tar = self.q_values[(new_state, best_action)]

                action_value = (reward + self.gamma * q_tar - q_now)  # (r(x, a) + lambda * max_b(Q_n(Y_n+1(x,a), b)) - Q_n(x, a))
                action_value = q_now + self.alp * action_value  # Q_n(x, a) + action_value

                self.q_values[(self.state, action)] = action_value
                self.state = self.env.reset() if is_done else new_state  # reset or run from current state
            is_done = False

    def eval_play_game(self, env: gym.envs) -> float:
        """
        Function used to evaluate the given policy by playing the game for an entire episode.

        :param env (gym.envs): The environment to play the game in.
        :return (float): The total reward of the play through.
        """

        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)  # select the best action - the one with best value
            new_state, reward, is_done, _ = env.step(action)  # iterate in the game
            self.rewards[(state, action, new_state)] = reward  # set reward in rewards table
            self.transits[(state, action)][new_state] += 1  # append to transit table
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward  # get the total reward of the play though.

    def eval_play_game_all(self, env: gym.envs) -> float:
        """
        Function used to evaluate the given policy by playing all combinations of the games.

        :param env (gym.envs): The environment to play the game in.
        :return (float): The total reward of the play through.
        """

        gen_all_states = self.env.iter_all_states()  # get all the states
        gen_all_states = iter(gen_all_states)

        total_reward = 0.0
        games = 0
        for st in gen_all_states:  # for every state in the game
            state = env.reset_for_testing_agent(state=st)  # reset with the current state
            games += 1
            while True:
                action = self.select_action(state)  # select the best action - the one with best value
                new_state, reward, is_done, _ = env.step(action)  # iterate in the game
                self.rewards[(state, action, new_state)] = reward  # set reward in rewards table
                self.transits[(state, action)][new_state] += 1  # append to transit table
                total_reward += reward
                if is_done:
                    break
                state = new_state

        avg_reward = total_reward / games
        return avg_reward


def Q_learn_opt(params: dict):
    start = time.time()
    reward_eval_que = collections.deque(maxlen=params["amount_of_eval_rounds"])
    test_env = chest_env(number_of_actions=params["number_of_actions"], grid_size=params["grid_size"], normalize=False,
                         return_distance=False, use_tensor=False)
    agent = Agent(env=test_env, gamma=params["gamma"], decay=params["decay"])

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.q_learn(nr_episodes=4 ** params["grid_size"], epsi=params["epsilon"])
        # fill in the Q-table - the future payments

        reward = 0.0
        if params["grid_size"] > 4:
            for _ in range(params["test_episodes"]):  # test the policy now for TEST_EPISODES games
                reward += agent.eval_play_game(test_env)
                reward /= params["test_episodes"]  # get the average reward
        else:
            reward = agent.eval_play_game_all(test_env)

        reward_eval_que.append(reward)
        # print("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        m_reward = torch.stack(list(reward_eval_que), dim=0).mean()
        if m_reward > 0.9999:
            print("Solved in %d iterations!" % iter_no)
            break

        if iter_no > (14 - params["grid_size"]) ** params["grid_size"]:
            # the parameter setting could not solve the env.
            break

    end = time.time()
    total_time = end - start
    return total_time, iter_no

def Q_learn_main(params: dict):
    start = time.time()
    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="ReL/ReLe-final",
        api_token=token,
    )

    reward_eval_que = collections.deque(maxlen=params["amount_of_eval_rounds"])

    run["algo"] = "Q_learning"

    run["parameters"] = params
    test_env = chest_env(number_of_actions=params["number_of_actions"], grid_size=params["grid_size"], normalize=False,
                         return_distance=False, use_tensor=False)
    agent = Agent(env=test_env, gamma=params["gamma"], decay=params["decay"])

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        # agent.play_n_random_steps(300)  # fill in the transit and rewards tables
        agent.q_learn(nr_episodes=4 ** params["grid_size"], epsi=params["epsilon"])
        # fill in the Q-table - the future payments

        run["alpha"].log(agent.alp)

        reward = 0.0
        if params["grid_size"] > 4:
            for _ in range(params["test_episodes"]):  # test the policy now for TEST_EPISODES games
                reward += agent.eval_play_game(test_env)
                reward /= params["test_episodes"]  # get the average reward
        else:
            reward = agent.eval_play_game_all(test_env)

        reward_eval_que.append(reward)
        run["reward"].log(reward)
        # print("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        m_reward = torch.stack(list(reward_eval_que), dim=0).mean()
        if m_reward > 0.9999:
            run["agent/rewards_size"].log(len(agent.rewards))
            run["agent/transits_size"].log(len(agent.transits))
            run["agent/values_size"].log(len(agent.q_values))
            # breakpoint()
            print("Solved in %d iterations!" % iter_no)
            break

    end = time.time()
    total_time = end - start
    run["time"] = total_time
    run["iter_no"] = iter_no
    run_id = run["sys/id"].fetch()
    run.stop()
    return total_time, iter_no, run_id

if __name__ == "__main__":

    params = {"number_of_actions": 20,
              "grid_size": 3,
              "epsilon": 0.2,
              "gamma": 0.9,
              "decay": 0.01,
              "test_episodes": 20,
              "amount_of_eval_rounds": 100}

    Q_learn_main(params)
