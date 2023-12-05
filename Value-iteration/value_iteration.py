import gym
import collections

from rele_pack.chest_env import chest_env

GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = chest_env(number_of_actions=10, grid_size=5)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)  # future values of the states - the one's we discount back

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
        """
        target_counts = self.transits[(state, action)]  # for the current state + action
        total = sum(target_counts.values())  # total = C_1 + C_2 + ... + C_n for n states
        action_value = 0.0
        for tgt_state, count in target_counts.items():  # eg. Counter({0: 12, 4: 3})
            reward = self.rewards[(state, action, tgt_state)]  # Hvad er reward for at være i denne state?
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state):
        """
        Based on the state that we are in select the best action (the one with best value).
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):  # all possible actions
            action_value = self.calc_action_value(state, action)  # get the values of said actions
            if best_value is None or best_value < action_value:  # get the best action
                best_value = action_value
                best_action = action
        return best_action

    def value_iteration(self):
        for state in range(self.env.observation_space.n):  # for every state in the game
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            # iterate over all the possible actions.

            self.values[state] = max(state_values)  # tag en value som maksimere vores state_values

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
    test_env = chest_env(number_of_actions=15, grid_size=5)
    agent = Agent()

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)  # fill in the transit and rewards tables
        agent.value_iteration()  # fill in the values table - the future payments

        reward = 0.0
        for _ in range(TEST_EPISODES):  # test the policy now for TEST_EPISODES games
            reward += agent.eval_play_game(test_env)

        reward /= TEST_EPISODES  # get the average reward
        print("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break

