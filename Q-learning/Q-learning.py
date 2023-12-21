import sys
sys.path.append('..')  # Add the parent directory to sys.path
sys.path.append('../..')  # Add the parent directory to sys.path
import gym
import collections
from rele_pack.chest_env import chest_env
import torch
import neptune
import os

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
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.q_values = collections.defaultdict(float)  # table with Q-values

        it = self.sqd()
        self.alpha = lambda: next(it)  # get the next alpha value in the sequence

    def sqd(self, lr=1.0):
        i = 1
        while True:
            yield lr / (i*0.01 + 1)
            i += 1

    # def play_n_random_steps(self, count):
    #     for _ in range(count):
    #         action = self.env.sample()  # takes random action
    #         new_state, reward, is_done, _ = self.env.step(action)  # get reward, and new_state
    #         self.rewards[(self.state, action, new_state)] = reward  # insert in reward table
    #         self.transits[(self.state, action)][new_state] += 1  # increment with 1 in the transaction table
    #         self.state = self.env.reset() if is_done else new_state  # reset or run from current state

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

                action_value = (reward + LAMBDA * q_tar - q_now)  # (r(x, a) + lambda * max_b(Q_n(Y_n+1(x,a), b)) - Q_n(x, a))
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



if __name__ == "__main__":
    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="ReL/ReLe-final",
        api_token=token,
    )



    params = {"number_of_actions": 20,
              "grid_size": 4,
              "epsilon": 0.2,
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
        # agent.play_n_random_steps(300)  # fill in the transit and rewards tables
        agent.q_learn(nr_episodes=30, epsi=params["epsilon"])  # fill in the values table - the future payments

        run["alpha"].log(agent.alp)

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
            # breakpoint()
            print("Solved in %d iterations!" % iter_no)
            break
