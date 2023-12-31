from Q_learning import Q_learn_main
import os
import neptune
import numpy as np

if __name__ == "__main__":
    params = {"gamma": 0.9,
              "test_episodes": 20,
              "amount_of_eval_rounds": 100}

    param_to_try = {(g, n): {"time": [], "it_number": [], "run_id": []} for g, n in
                    zip([2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [5, 9, 12, 20, 30, 40, 45, 50, 60])}

    for grid_size, number_of_actions in param_to_try:
        for _ in range(5):
            params_to_try = params | {"number_of_actions": number_of_actions, "grid_size": grid_size}

            if grid_size == 2:
                params_to_try = params_to_try | {"epsilon": 0.4977691437307691, "decay": 0.09914277382720374}
            elif grid_size == 3:
                params_to_try = params_to_try | {"epsilon": 0.4727390247809401, "decay": 0.050325305990524295}
            elif grid_size == 4:
                params_to_try = params_to_try | {"epsilon": 0.49418818387616764, "decay": 0.09211683295961771}
            else:
                params_to_try = params_to_try | {"epsilon": 0.5, "decay": 0.06}

            t, it_no, run_id = Q_learn_main(params_to_try)
            param_to_try[(grid_size, number_of_actions)]["time"].append(t)
            param_to_try[(grid_size, number_of_actions)]["it_number"].append(it_no)
            param_to_try[(grid_size, number_of_actions)]["run_id"].append(run_id)

        token = os.getenv('NEPTUNE_API_TOKEN')
        run = neptune.init_run(
            project="ReL/ReLe-final-results",
            api_token=token,
            with_id="REL1-6",
        )

        # run["algo"] = "Q_learn"

        param_to_try[(grid_size, number_of_actions)]["time"]\
            .append(np.mean(param_to_try[(grid_size, number_of_actions)]["time"]))
        param_to_try[(grid_size, number_of_actions)]["it_number"]\
            .append(np.mean(param_to_try[(grid_size, number_of_actions)]["it_number"]))

        run[f"time/time_Q_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["time"])
        run[f"it_number/it_no_Q_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["it_number"])
        run[f"run_id/run_Q_grid_{grid_size}_nac_{number_of_actions}"]\
            = str(param_to_try[(grid_size, number_of_actions)]["run_id"])

        run.stop()
