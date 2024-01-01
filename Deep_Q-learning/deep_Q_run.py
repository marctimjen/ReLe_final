from deep_Q import deep_q_main
import os
import neptune
import numpy as np


if __name__ == "__main__":
    params = {"gamma": 0.99,
              "batch_size": 32,
              "replay_size": 2000000,
              "lr": 1e-4,
              "replay_start_size": 50000,
              "epsilon_start": 0.99,
              "epsilon_final": 0.000001,
              "amount_of_eval_rounds": 450,
              }

    param_to_try = {(g, n): {"time": [], "it_number": [], "run_id": []} for g, n in
                    zip([2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [5, 9, 12, 20, 30, 40, 45, 50, 60])}

    for grid_size, number_of_actions in param_to_try:
        for _ in range(5):
            params_to_try = params | {"number_of_actions": number_of_actions, "grid_size": grid_size}

            if grid_size == 2:
                params_to_try = params_to_try | {"sync_target_frames": 40000, "epsilon_decay_last_frame": 100000}
            elif grid_size == 3:
                params_to_try = params_to_try | {"sync_target_frames": 10000, "epsilon_decay_last_frame": 250000}
            elif grid_size == 4:
                params_to_try = params_to_try | {"sync_target_frames": 40000, "epsilon_decay_last_frame": 500000}
            else:
                params_to_try = params_to_try | {"sync_target_frames": 50000,
                                                 "epsilon_decay_last_frame": 250000 * grid_size,
                                                 "replay_size": 3500000,
                                                 "replay_start_size": 70000}

            t, it_no, run_id = deep_q_main(params_to_try)
            param_to_try[(grid_size, number_of_actions)]["time"].append(t)
            param_to_try[(grid_size, number_of_actions)]["it_number"].append(it_no)
            param_to_try[(grid_size, number_of_actions)]["run_id"].append(run_id)

        token = os.getenv('NEPTUNE_API_TOKEN')
        run = neptune.init_run(
            project="ReL/ReLe-final-results",
            api_token=token,
            with_id="REL1-7",
        )

        # run["algo"] = "Deep_Q_learning"

        param_to_try[(grid_size, number_of_actions)]["time"]\
            .append(np.mean(param_to_try[(grid_size, number_of_actions)]["time"]))
        param_to_try[(grid_size, number_of_actions)]["it_number"]\
            .append(np.mean(param_to_try[(grid_size, number_of_actions)]["it_number"]))

        run[f"time/time_deep_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["time"])
        run[f"it_number/it_no_deep_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["it_number"])
        run[f"run_id/run_deep_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["run_id"])

        run.stop()
