import optuna
import neptune.integrations.optuna as optuna_utils
import neptune
import os
import argparse

parser = argparse.ArgumentParser(description='opt_test')
parser.add_argument("-r", "--run", required=True, help="run id")
args = parser.parse_args()


token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token,
    with_id=args.run
)  # your credentials

params = {"direction": "minimize", "n_trials": 20}

def objective(trial):
    param = {
        "epochs": trial.suggest_int("epochs", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.8),
    }

    loss = (param["dropout"] * param["learning_rate"]) ** param["epochs"]

    return loss


neptune_callback = optuna_utils.NeptuneCallback(run)

study = optuna.load_study(study_name="Q_learn_study", storage="sqlite:///example.db")
study.optimize(objective, n_trials=params["n_trials"], callbacks=[neptune_callback])

run.stop()
