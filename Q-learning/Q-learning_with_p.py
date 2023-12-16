import sys
sys.path.append('..')  # Add the parent directory to sys.path
sys.path.append('../..')  # Add the parent directory to sys.path
import gym
import collections
from rele_pack.chest_env import chest_env
import torch
import neptune
import os

# 0: (0, 0) # stand still
# 1: (0, 1) # 1 down
# 2: (1, 0) # 1 right
# 3: (1, 1) # 1 right and 1 down
# 4: (0, -1) # 1 up
# 5: (-1, 0) # 1 left
# 6: (-1, -1) # 1 left and 1 up
# 7: (-1, 1) # 1 left and 1 down
# 8: (1, -1) # 1 right and 1 up


"""
This implementation of the Q-learning algorithm estimate the probabilites: p(y |x, a) directly.
This means that the Q-update step looks alto more like the value iteration update with;

Q = reward + lambda * sum_y max_a(Q(y, a)) * p(y|x, a)
"""


class Agent:
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.q_values = collections.defaultdict(float)  # table with Q-values

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.sample()  # takes random action
            new_state, reward, is_done, _ = self.env.step(action)  # get reward, and new_state
            self.rewards[(self.state, action, new_state)] = reward  # insert in reward table
            self.transits[(self.state, action)][new_state] += 1  # increment with 1 in the transaction table
            self.state = self.env.reset() if is_done else new_state  # reset or run from current state

    def select_action(self, state):
        """
        Based on the state that we are in select the best action (the one with best value).
        """
        best_action, best_value = None, None

        gen_all_actions = self.env.iter_all_actions()  # get all the actions

        for action in gen_all_actions:  # all possible actions
            action_value = self.q_values[(state, action)]  # take the action that maximize the Q-function
            if best_value is None or best_value < action_value:  # get the best action
                best_value = action_value
                best_action = action
        return best_action

    def q_learn(self):
        """
        Here we iterate over all states and actions and calculate the max state value. With this we can fill the values
        table.

        :return: None
        """
        gen_all_states = self.env.iter_all_states()  # get all the states
        gen_all_states = iter(gen_all_states)

        for state in gen_all_states:  # for every state in the game
            gen_all_actions = self.env.iter_all_actions()  # get all the actions
            gen_all_actions = iter(gen_all_actions)
            for action in gen_all_actions:
                action_value = 0.0
                target_counts = self.transits[(state, action)]  # get the amount of transits for this state and action
                total = sum(target_counts.values())  # C_1 + ... + C_n
                for tgt_state, count in target_counts.items():  # For every target state we can end up in and the times we have done so
                    key = (state, action, tgt_state)  # get key
                    reward = self.rewards[key]  # get rewards for going from state to tgt_state using action
                    best_action = self.select_action(tgt_state)  # select the best action to take
                    val = reward + LAMBDA * self.q_values[(tgt_state, best_action)]  # val = r(x, a) + lambda * max_a(Q(Y_n+1, a))
                    action_value += (count / total) * val  # action_value = p(y|x, a) * val
                self.q_values[(state, action)] = action_value  # update the values table using r(x, a) + lambda * sum_y p(y|x, a) * V^n(y)

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



if __name__ == "__main__":
    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="ReL/ReLe-final",
        api_token=token,
    )

    run["Algo"] = "Q-learning"

    params = {"number_of_actions": 20,
              "grid_size": 3,
              "lambda": 0.9,
              "test_episodes": 20,
              "amount_of_eval_rounds": 100}

    reward_eval_que = collections.deque(maxlen=params["amount_of_eval_rounds"])

    LAMBDA = params["lambda"]
    TEST_EPISODES = params["test_episodes"]

    run["parameters"] = params
    test_env = chest_env(number_of_actions=params["number_of_actions"], grid_size=params["grid_size"], normalize=False,
                         return_distance=False, use_tensor=False)
    agent = Agent(env=test_env)

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(300)  # fill in the transit and rewards tables
        agent.q_learn()  # fill in the values table - the future payments

        reward = 0.0
        for _ in range(TEST_EPISODES):  # test the policy now for TEST_EPISODES games
            reward += agent.eval_play_game(test_env)

        reward /= TEST_EPISODES  # get the average reward
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
            run.stop()
            #breakpoint()
            print("Solved in %d iterations!" % iter_no)
            break
