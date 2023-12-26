import optuna
from optuna.samplers import TPESampler
import neptune
import neptune.integrations.optuna as optuna_utils
import os
from Q_learning import Q_learn_opt

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token,
)

run["Algo"] = "Q_learning"
run["grid_size"] = "2"

neptune_callback = optuna_utils.NeptuneCallback(run)


def objective(trial):
    opt_id = trial.number

    params = {"number_of_actions": 20,
              "grid_size": 2,
              "epsilon": 0.2,
              "gamma": 0.9,
              "decay": trial.suggest_float("gamma", 0.000000001, 0.05),
              "test_episodes": 20,
              "amount_of_eval_rounds": 100}

    time, iter_no = Q_learn_opt(params)

    run[f"trials/trials/{opt_id}/reward/time"].log(time)
    run[f"trials/trials/{opt_id}/reward/iter_no"].log(iter_no)
    return time


study = optuna.create_study(sampler=TPESampler(), direction="minimize")
study.optimize(objective, n_trials=500, callbacks=[neptune_callback])

run.stop()
