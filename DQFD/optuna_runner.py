import time
import subprocess
import neptune.integrations.optuna as optuna_utils
import neptune
import os
import optuna
import argparse
from dqfd import dqfd_main

parser = argparse.ArgumentParser(description='dqfd optimization')
parser.add_argument("-g", "--grid", required=True, help="grid-size")
parser.add_argument("-n", "--act", required=True, help="number_of_actions")
args = parser.parse_args()

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token,
    with_id=args.run
)  # your credentials

run["Algo"] = "DQfD"
run["grid_size"] = args.grid
run["number_of_actions"] = args.act
run_id = run["sys/id"].fetch()

lambda_loss = {"lambda_dq": 1, "lambda_n": 1, "lambda_je": 1, "lambda_l2": 0.0005}
params = {"number_of_actions": int(args.act),
          "grid_size": int(args.grid),
          "gamma": 0.99,
          "batch_size": 32,
          "replay_size": 200000,
          "lr": 1e-4,
          "sync_target_frames": "to_be_tuned",
          "replay_start_size": 50000,
          "epsilon_decay_last_frame": "to_be_tuned",
          "epsilon_start": 0.99,
          "epsilon_final": 0.000001,
          "beta_frames": "to_be_tuned",
          "beta_start": 0.4,
          "expert_play": "to_be_tuned",  # The amount of expert frames!
          "pre_train_phase": "to_be_tuned",
          "lambda_loss": lambda_loss,  # the weights for the different loss-functions
          "amount_of_eval_rounds": 450,  # how many games to ace in a row.
          }

run["parameters"] = params


if f"DQfD_learn_study_grid_{args.grid}_act_{args.act}" in optuna.study.get_all_study_names("sqlite:///example.db"):
    optuna.delete_study(study_name=f"DQfD_learn_study_grid_{args.grid}_act_{args.act}", storage="sqlite:///example.db")

study = optuna.create_study(storage="sqlite:///example.db", study_name=f"DQfD_learn_study_grid_{args.grid}_act_{args.act}",
                            direction="minimize")

neptune_callback = optuna_utils.NeptuneCallback(run)

def objective(trial):
    opt_id = trial.number

    if int(args.grid) == 2:
        epsilon_decay = trial.suggest_categorical('epsilon_decay', [10000, 25000, 40000])
        expert_play = trial.suggest_categorical('expert_play', [5000, 10000, 40000])
        beta_frames = trial.suggest_categorical('beta_frames', [50000, 100000, 250000])
    elif int(args.grid) == 3:
        epsilon_decay = trial.suggest_categorical('epsilon_decay', [25000, 50000, 100000])
        expert_play = trial.suggest_categorical('expert_play', [10000, 25000, 40000])
        beta_frames = trial.suggest_categorical('beta_frames', [50000, 100000, 250000])
    elif int(args.grid) == 4:
        epsilon_decay = trial.suggest_categorical('epsilon_decay', [50000, 100000, 500000])
        expert_play = trial.suggest_categorical('expert_play', [25000, 40000, 60000])
        beta_frames = trial.suggest_categorical('beta_frames', [100000, 250000, 500000])
    else:
        raise ValueError("Should not have input for grid size 2-4")

    lambda_loss = {"lambda_dq": 1, "lambda_n": 1, "lambda_je": 1, "lambda_l2": 0.0005}
    params = {"number_of_actions": int(args.act),
              "grid_size": int(args.grid),
              "gamma": 0.99,
              "batch_size": 32,
              "replay_size": 200000,
              "lr": 1e-4,
              "sync_target_frames": trial.suggest_categorical('sync_target_frames', [20_000, 40_000]),
              "replay_start_size": 50000,
              "epsilon_decay_last_frame": epsilon_decay,
              "epsilon_start": 0.99,
              "epsilon_final": 0.000001,
              "beta_frames": beta_frames,
              "beta_start": 0.4,
              "expert_play": expert_play,  # The amount of expert frames!
              "pre_train_phase": int(expert_play/5),
              "lambda_loss": lambda_loss,  # the weights for the different loss-functions
              "amount_of_eval_rounds": 450,  # how many games to ace in a row.
              }

    time_t, iter_no, _ = dqfd_main(params)

    run[f"trials/trials/{opt_id}/reward/time"].log(time_t)
    run[f"trials/trials/{opt_id}/reward/iter_no"].log(iter_no)
    return iter_no


n_amount = 10
study.optimize(objective, n_trials=n_amount, callbacks=[neptune_callback])
run.stop()
