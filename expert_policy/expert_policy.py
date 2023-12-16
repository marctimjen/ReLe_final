import sys
sys.path.append('..')  # Add the parent directory to sys.path
sys.path.append('../..')  # Add the parent directory to sys.path
from rele_pack.chest_env import chest_env
from rele_pack.expert_policy import expert_policy

if __name__ == '__main__':
    params = {"number_of_actions": 10,
              "grid_size": 5,
              "amount_of_eval_rounds": 100}

    env = chest_env(number_of_actions=params["number_of_actions"], grid_size=params["grid_size"], normalize=True,
                    return_distance=True, use_tensor=True)
    obs = env.reset()

    print(env.player_position)
    print(env.coin_position)
    print(obs)

    exp_pol = expert_policy()

    is_done = False
    frame_idx = 0

    while not(is_done):
        frame_idx += 1

        action = exp_pol.play_step(obs)
        print("frame index =", frame_idx, "  action =", action)

        obs, reward, is_done, _ = env.step(action)

        print("Player pos:", env.player_position)
        print("coin pos:", env.coin_position)
        print("chest pos:", env.chest_position)

        print(obs)
        print(reward)
