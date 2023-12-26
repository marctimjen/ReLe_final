import optuna
import neptune
import neptune.integrations.optuna as optuna_utils
import os
from Q_learning import Q_learn_opt
import argparse

parser = argparse.ArgumentParser(description='Q-learn optimization')
parser.add_argument("-r", "--run", required=True, help="run id")
parser.add_argument("-g", "--grid", required=True, help="grid-size")
parser.add_argument("-n", "--act", required=True, help="number_of_actions")
args = parser.parse_args()

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token,
    with_id=args.run
)  # your credentials

neptune_callback = optuna_utils.NeptuneCallback(run)


def objective(trial):
    opt_id = trial.number

    params = {"number_of_actions": int(args.act),
              "grid_size": int(args.grid),
              "epsilon": trial.suggest_float("epsilon", 0.05, 0.5),
              "gamma": 0.9,
              "decay": trial.suggest_float("decay", 0.000000001, 0.1),
              "test_episodes": 20,
              "amount_of_eval_rounds": 100}

    time, iter_no = Q_learn_opt(params)

    run[f"trials/trials/{opt_id}/reward/time"].log(time)
    run[f"trials/trials/{opt_id}/reward/iter_no"].log(iter_no)
    return iter_no

study = optuna.load_study(study_name=f"Q_learn_study_grid_{args.grid}_act_{args.act}", storage="sqlite:///example.db")
study.optimize(objective, n_trials=10, callbacks=[neptune_callback])
run.stop()
