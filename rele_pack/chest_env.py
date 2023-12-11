import gym
import torch
from gym import spaces

# (right = 1 or left = -1, down = 1 or up = -1)
# 0: (0, 0) # stand still
# 1: (0, 1) # 1 down
# 2: (1, 0) # 1 right
# 3: (1, 1) # 1 right and 1 down
# 4: (0, -1) # 1 up
# 5: (-1, 0) # 1 left
# 6: (-1, -1) # 1 left and 1 up
# 7: (-1, 1) # 1 left and 1 down
# 8: (1, -1) # 1 right and 1 up

device = torch.device("cpu")  # "cuda:0" if torch.cuda.is_available() else
fl = torch.cuda.FloatTensor
class chest_env(gym.Env):
    def __init__(self, number_of_actions=256, grid_size=128, normalize=True, return_distance=True, use_tensor=True):
        """
        :param distance: Should be the distance between the player and the coin
        """
        super(chest_env, self).__init__()
        self.normalize = normalize
        self.return_distance = return_distance
        self.use_tensor = use_tensor
        self.grid_size = torch.tensor((grid_size, grid_size), dtype=torch.int, device=device)  # change this to set the size of the environment
        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))  # Four discrete actions: up, down, left, right
        self.observation_space = spaces.Tuple((spaces.Discrete(grid_size), spaces.Discrete(grid_size)))  # State space: (x, y)
        self.player_position = None
        self.coin_position = None
        self.number_of_actions = torch.tensor((number_of_actions), dtype=torch.int, device=device)
        self.action_number = torch.tensor((0), dtype=torch.int, device=device)


    def reset(self):
        pl_x = torch.randint(0, self.grid_size[0], size=(1, 1), dtype=torch.int, device=device).flatten()
        pl_y = torch.randint(0, self.grid_size[1], size=(1, 1), dtype=torch.int, device=device).flatten()
        self.player_position = torch.concat((pl_x, pl_y))
        # self.player_position = [0, 0]  # Start at the top-left corner
        self.coin_position = self._place_item()
        self.chest_position = self._place_item()

        self.has_coin = torch.tensor((False), dtype=torch.bool, device=device)
        self.action_number = 0

        observation = self.return_observation()
        return observation

    def return_observation(self):
        if self.normalize:
            self.player_return = self.normalize(self.player_position)
            self.coin_return = self.normalize(self.coin_position)
            self.chest_return = self.normalize(self.chest_position)
        else:
            self.player_return = self.player_position
            self.coin_return = self.coin_position
            self.chest_return = self.chest_position


        if self.return_distance:
            observation = torch.concat((self.player_return, self.coin_return, self.chest_return,
                                        self.coin_return - self.player_return,
                                        self.chest_return - self.player_return)).double()
        else:
            observation = torch.concat((self.player_return, self.coin_return, self.chest_return)).double()

        if not self.use_tensor:
            observation = observation.numpy()

            if not self.normalize:
                observation = observation.astype(int)

            observation = tuple(observation)

        return observation

    def dist(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        return torch.sqrt(torch.sum((x - y) ** 2))

    def step(self, action):

        h, v = action
        x, y = self.player_position
        # dist_before = self.dist(self.player_position, self.coin_position)

        if h == 0:
            # Move left
            y = max(torch.tensor((0), dtype=torch.int, device=device), y - 1)
        elif h == 2:
            # Move right
            y = min(self.grid_size[1] - 1, y + 1)
        # else:  the agent does not move!

        if v == 0:
            # Move up
            x = max(torch.tensor((0), dtype=torch.int, device=device), x - 1)
        elif v == 2:
            # Move down
            x = min(self.grid_size[0] - 1, x + 1)
        # else: the agent does not move!

        self.player_position = torch.concat((x[None], y[None]))

        if self.has_coin:
            self.coin_position = self.player_position

        # dist_after = self.dist(self.player_position, self.coin_position)
        self.action_number += 1

        # Check if the player has collected the coin
        if torch.eq(self.player_position, self.coin_position).all() and not self.has_coin:
            reward = torch.tensor((0.5), dtype=torch.float32, device=device)  # Provide a positive reward for collecting the coin
            self.has_coin = torch.tensor((True), dtype=torch.bool, device=device)
            done = torch.tensor((False), dtype=torch.bool, device=device)
        elif torch.eq(self.player_position, self.coin_position).all() and torch.eq(self.player_position, self.chest_position).all():
            reward = torch.tensor((0.5), dtype=torch.float32, device=device)  # Provide a positive reward for collecting the coin
            self.has_coin = torch.tensor((False), dtype=torch.bool, device=device)
            done = torch.tensor((True), dtype=torch.bool, device=device)
        elif self.action_number < self.number_of_actions:

            reward = torch.tensor((0), dtype=torch.float32, device=device)
            done = torch.tensor((False), dtype=torch.bool, device=device)
        else:
            reward = torch.tensor((0), dtype=torch.float32, device=device)
            done = torch.tensor((True), dtype=torch.bool, device=device)

        observation = self.return_observation()
        return observation, reward, done, {}

    def render(self):
        # Implement visualization (optional)
        pass

    def sample(self):

        sampled_action = torch.randint(0, 3, size=(2, 1), dtype=torch.int, device=device)

        if not self.use_tensor:
            sampled_action = sampled_action.numpy()
            sampled_action = tuple(sampled_action.flatten())

        return sampled_action

    def normalize(self, x: list):
        return x/self.grid_size

    def _place_item(self):
        x = torch.randint(0, self.grid_size[0], size=(1, 1), dtype=torch.int, device=device).flatten()
        y = torch.randint(0, self.grid_size[1], size=(1, 1), dtype=torch.int, device=device).flatten()

        return torch.concat((x, y))

    def iter_all_states(self):
        """
        This function will return an iterator, that can loop over all the states for this game :).
        This is usefull for the implementation of the value-iteration algorithm.
        :return: iterator
        """


        def it_states(grid_size, device=device, use_tensor=True):
            for x in range(grid_size[0]):
                for y in range(grid_size[1]):
                    for x2 in range(grid_size[0]):
                        for y2 in range(grid_size[1]):
                            for x3 in range(grid_size[0]):
                                for y3 in range(grid_size[1]):

                                    if use_tensor:
                                        yield torch.tensor((x, y, x2, y2, x3, y3), dtype=torch.int, device=device)
                                    else:
                                        yield (x, y, x2, y2, x3, y3)

        it = it_states(grid_size=self.grid_size, device=device, use_tensor=self.use_tensor)

        return it

    def iter_all_actions(self):
        """
        This function will return an iterator, that can loop over all the states for this game :).
        This is usefull for the implementation of the value-iteration algorithm.

        :return: iterator
        """

        def it_actions(device=device, use_tensor=True):
            for x in range(3):
                for y in range(3):
                    if use_tensor:
                        yield torch.tensor((x, y), dtype=torch.int, device=device)
                    else:
                        yield (x, y)

        it = it_actions(device=device, use_tensor=self.use_tensor)
        return it






if __name__ == '__main__':
    gym.register(id='chest_env-v0', entry_point='chest_env:chest_env')