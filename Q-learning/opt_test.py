import optuna
import dask.distributed
from dask.distributed import Client
from dask_optuna import OptunaScheduler
import neptune
import neptune.integrations.optuna as optuna_utils
import os

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token
)  # your credentials

params = {"direction": "minimize", "n_trials": 20}
run["parameters"] = params


def objective(trial):
    param = {
        "epochs": trial.suggest_int("epochs", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.2, 0.8),
    }

    loss = (param["dropout"] * param["learning_rate"]) ** param["epochs"]

    return loss


neptune_callback = optuna_utils.NeptuneCallback(run)

client = Client()
scheduler = OptunaScheduler(client)
study = optuna.create_study(direction=params["direction"], scheduler=scheduler)
study.optimize(objective, n_trials=params["n_trials"], callbacks=[neptune_callback])

run.stop()
