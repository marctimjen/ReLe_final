import optuna
import dask.distributed
from dask.distributed import Client
from dask_optuna import OptunaScheduler
import neptune
import neptune.integrations.optuna as optuna_utils
import os
import joblib

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

with dask.distributed.Client() as client:
   # Create a study using Dask-compatible storage
   storage = dask_optuna.DaskStorage()
   study = optuna.create_study(storage=storage, direction=params["direction"])
   # Optimize in parallel on your Dask cluster
   with joblib.parallel_backend("dask"):
      study.optimize(objective, n_trials=params["n_trials"], callbacks=[neptune_callback], n_jobs=-1)
   print(f"best_params = {study.best_params}")

run.stop()
