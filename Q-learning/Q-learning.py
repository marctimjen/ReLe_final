import gym
import numpy as np
import sys
sys.path.append('..')  # Add the parent directory to sys.path
sys.path.append('../..')  # Add the parent directory to sys.path
from rele_pack.chest_env import chest_env
import os
import neptune
# token = os.getenv('NEPTUNE_API_TOKEN')
# run = neptune.init_run(
#     project="Kernel-bois/reinforcement-learning",
#     api_token=token,
# )





# 0: (0, 0) # stand still
# 1: (0, 1) # 1 down
# 2: (1, 0) # 1 right
# 3: (1, 1) # 1 right and 1 down
# 4: (0, -1) # 1 up
# 5: (-1, 0) # 1 left
# 6: (-1, -1) # 1 left and 1 up
# 7: (-1, 1) # 1 left and 1 down
# 8: (1, -1) # 1 right and 1 up


if __name__ == '__main__':
    env = chest_env()  # load the environment
    env.reset()  # reset the environment

print()



