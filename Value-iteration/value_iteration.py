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


# (right = 2 or left = 0, down = 2 or up = 0)
# 0: (0, 0) # Move up and left
# 1: (0, 1) # Move up
# 2: (1, 0) # Move left
# 3: (1, 1) # stand still
# 4: (0, 2) # Move up and right
# 5: (2, 0) # Move down and left
# 6: (1, 2) # Move right
# 7: (2, 1) # Move down
# 8: (2, 2) # Move down and right


class Agent:
    def __init__(self, env, gamma: float):
        self.env = env
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)  # future values of the states - the one's we discount back
        self.gamma = gamma

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.sample()  # takes random action
            new_state, reward, is_done, _ = self.env.step(action)  # get reward, and new_state
            self.rewards[(self.state, action, new_state)] = reward  # insert in reward table
            self.transits[(self.state, action)][new_state] += 1  # increment with 1 in the transaction table
            self.state = self.env.reset() if is_done else new_state  # reset or run from current state

    def calc_action_value(self, state, action):
        """
        Given a state and the action return the value of this combination.
        For the given state and action return r(x,a) + gamma * sum_y p(y|x, a) * V^n(y)
        """
        target_counts = self.transits[(state, action)]  # for the current state + action
        total = sum(target_counts.values())  # total = C_1 + C_2 + ... + C_n for n states
        action_value = torch.tensor(0.0)
        for tgt_state, count in target_counts.items():  # eg. Counter({0: 12, 4: 3})
            reward = self.rewards[(state, action, tgt_state)]  # Hvad er reward for at være i denne state?
            val = reward + self.gamma * self.values[tgt_state]  # val = r(x, a) + gamma * V^n(y)
            action_value += (count / total) * val  # action_value =  p(y|x, a) * val
        return action_value  # sum_y action_value  => for the given state and action return r(x,a) + gamma * sum_y p(y|x, a) * V^n(y)


    def select_action(self, state):
        """
        Based on the state that we are in select the best action (the one with best value).
        """
        best_action, best_value = None, None

        gen_all_actions = self.env.iter_all_actions()  # get all the actions

        for action in gen_all_actions:  # all possible actions
            action_value = self.calc_action_value(state, action)  # get the values of said actions
            if best_value is None or best_value < action_value:  # get the best action
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
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

            state_values = [self.calc_action_value(state, action) for action in gen_all_actions]
            # iterate over all the possible actions:  r(x, a) + gamma * sum_y p(y|x, a) * V^n(y)

            self.values[state] = max(state_values)
            # V^(n+1)(x) = max_a (r(x, a) + gamma * sum_y p(y|x, a)*V^n(y))


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


def value_it_main(params: dict):

    start = time.time()
    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="ReL/ReLe-final",
        api_token=token,
    )

    reward_eval_que = collections.deque(maxlen=params["amount_of_eval_rounds"])

    run["algo"] = "Value_iteration"
    run["parameters"] = params
    test_env = chest_env(number_of_actions=params["number_of_actions"], grid_size=params["grid_size"], normalize=False,
                         return_distance=False, use_tensor=False)
    agent = Agent(env=test_env, gamma=params["gamma"])

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(6 ** params["grid_size"])  # fill in the transit and rewards tables
        agent.value_iteration()  # fill in the values table - the future payments
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
            run["agent/values_size"].log(len(agent.values))
            break

    end = time.time()
    total_time = end - start
    run["time"] = total_time
    run["iter_no"] = iter_no
    run_id = run["sys/id"].fetch()
    run.stop()
    return total_time, iter_no, run_id


if __name__ == "__main__":
    params = {"number_of_actions": 25,
              "grid_size": 5,
              "gamma": 0.9,
              "test_episodes": 20,
              "amount_of_eval_rounds": 1000}

    value_it_main(params=params)
