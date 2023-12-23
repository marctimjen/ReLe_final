from value_iteration import value_it_main
import os
import neptune
import numpy as np


if __name__ == "__main__":
    params = {"gamma": 0.9,
              "test_episodes": 20,
              "amount_of_eval_rounds": 100}

    param_to_try = {(g, n): {"time": [], "it_number": []} for g, n in zip([3, 4], [9, 12])}

    for _ in range(2):
        for grid_size, number_of_actions in param_to_try:

            params_to_try = params | {"number_of_actions": number_of_actions, "grid_size": grid_size}
            t, it_no = value_it_main(params_to_try)
            param_to_try[(grid_size, number_of_actions)]["time"].append(t)
            param_to_try[(grid_size, number_of_actions)]["it_number"].append(it_no)

    token = os.getenv('NEPTUNE_API_TOKEN')
    run = neptune.init_run(
        project="ReL/ReLe-final-results",
        api_token=token,
    )

    run["algo"] = "Value_iteration"

    for g, n in param_to_try:
        param_to_try[(g, n)]["time"].append(np.mean(param_to_try[(g, n)]["time"]))
        param_to_try[(g, n)]["it_number"].append(np.mean(param_to_try[(g, n)]["it_number"]))
        run[f"time/time_value_grid_{g}_nac_{n}"] = str(param_to_try[(g, n)]["time"])
        run[f"it_number/it_no_value_grid_{g}_nac_{n}"] = str(param_to_try[(g, n)]["it_number"])

    run.stop()
