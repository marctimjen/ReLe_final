from dqfd import dqfd_opt
import os
import neptune
import numpy as np


if __name__ == "__main__":
    lambda_loss = {"lambda_dq": 1, "lambda_n": 1, "lambda_je": 1, "lambda_l2": 0.0005}

    params = {"gamma": 0.99,
              "batch_size": 32,
              "replay_size": 200000,
              "lr": 1e-4,
              "replay_start_size": 50000,
              "epsilon_start": 0.99,
              "epsilon_final": 0.000001,
              "beta_start": 0.4,
              "lambda_loss": lambda_loss,  # the weights for the different loss-functions
              "amount_of_eval_rounds": 450,  # how many games to ace in a row.
              }

    param_to_try = {(g, n): {"time": [], "it_number": [], "run_id": []} for g, n in
                    zip([2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [5, 9, 12, 20, 30, 40, 45, 50, 60])}

    for grid_size, number_of_actions in param_to_try:
        for _ in range(5):
            params_to_try = params | {"number_of_actions": number_of_actions, "grid_size": grid_size}

            if grid_size == 2:
                params_to_try = params_to_try | {"sync_target_frames": 40000, "epsilon_decay_last_frame": 40_000,
                                                 "expert_play": 5_000, "beta_frames": 250_000,
                                                 "pre_train_phase": int(5_000/5)}
            elif grid_size == 3:
                params_to_try = params_to_try | {"sync_target_frames": 20000, "epsilon_decay_last_frame": 50_000,
                                                 "expert_play": 10_000, "beta_frames": 50_000,
                                                 "pre_train_phase": int(10_000/5)}
            else:
                lambda_loss2 = {"lambda_dq": 1, "lambda_n": 1, "lambda_je": 1, "lambda_l2": 0.001}
                params_to_try = params_to_try | {"sync_target_frames": 40000,
                                                 "epsilon_decay_last_frame": 40000 + 1000 * (grid_size - 4),
                                                 "beta_frames": 50000 + 10000 * (grid_size - 4),
                                                 "expert_play": 10000 + 2500 * (grid_size - 4),  # The amount of expert frames!
                                                 "pre_train_phase": 10000 + 2500 * (grid_size - 4),
                                                 "lambda_loss": lambda_loss2}

            t, it_no, run_id = dqfd_opt(params_to_try)
            param_to_try[(grid_size, number_of_actions)]["time"].append(t)
            param_to_try[(grid_size, number_of_actions)]["it_number"].append(it_no)
            param_to_try[(grid_size, number_of_actions)]["run_id"].append(run_id)

        token = os.getenv('NEPTUNE_API_TOKEN')
        run = neptune.init_run(
            project="ReL/ReLe-final-results",
            api_token=token,
            with_id="REL1-8",
        )

        # run["algo"] = "DQfD"

        param_to_try[(grid_size, number_of_actions)]["time"]\
            .append(np.mean(param_to_try[(grid_size, number_of_actions)]["time"]))
        param_to_try[(grid_size, number_of_actions)]["it_number"]\
            .append(np.mean(param_to_try[(grid_size, number_of_actions)]["it_number"]))

        run[f"time/time_dqfd_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["time"])
        run[f"it_number/it_no_dqfd_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["it_number"])
        run[f"run_id/run_dqfd_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["run_id"])

        run.stop()
