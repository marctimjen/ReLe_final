import time
import subprocess
import neptune.integrations.optuna as optuna_utils
import neptune
import os
import optuna
import argparse
from deep_Q import deep_q_main

parser = argparse.ArgumentParser(description='Deep Q-learn start optimizers')
parser.add_argument("-g", "--grid", required=True, help="grid-size")
parser.add_argument("-n", "--act", required=True, help="number_of_actions")
args = parser.parse_args()

py_path = "/home/tyson/.conda/envs/rele/bin/python3"

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token
)  # your credentials

run["Algo"] = "Deep_Q_learning"
run["grid_size"] = args.grid
run["number_of_actions"] = args.act
run_id = run["sys/id"].fetch()

params = {"number_of_actions": int(args.act),
          "grid_size": int(args.grid),
          "gamma": 0.99,
          "batch_size": 32,
          "replay_size": 2000000,
          "lr": 1e-4,
          "sync_target_frames": "to_be_tuned",
          "replay_start_size": 50000,
          "epsilon_decay_last_frame": "to_be_tuned",
          "epsilon_start": 0.99,
          "epsilon_final": 0.000001,
          "amount_of_eval_rounds": 20 * 450,
          }

run["parameters"] = params


if f"Deep_Q_learn_study_grid_{args.grid}_act_{args.act}" in optuna.study.get_all_study_names("sqlite:///example.db"):
    optuna.delete_study(study_name=f"Deep_Q_learn_study_grid_{args.grid}_act_{args.act}", storage="sqlite:///example.db")

study = optuna.create_study(storage="sqlite:///example.db", study_name=f"Deep_Q_learn_study_grid_{args.grid}_act_{args.act}",
                            direction="minimize")

neptune_callback = optuna_utils.NeptuneCallback(run)


def objective(trial):
    opt_id = trial.number

    if int(args.grid) == 2:
        epsilon_decay = trial.suggest_categorical('epsilon_decay', [100_000, 250_000, 400_000])
    elif int(args.grid) == 3:
        epsilon_decay = trial.suggest_categorical('epsilon_decay', [250_000, 500_000, 1_000_000])
    elif int(args.grid) == 4:
        epsilon_decay = trial.suggest_categorical('epsilon_decay', [500_000, 1_000_000, 5_000_000])
    else:
        epsilon_decay = trial.suggest_categorical('epsilon_decay', [1_000_000, 5_000_000, 10_000_000])

    params = {"number_of_actions": int(args.act),
              "grid_size": int(args.grid),
              "gamma": 0.99,
              "batch_size": 32,
              "replay_size": 2000000,
              "lr": 1e-4,
              "sync_target_frames": trial.suggest_categorical('sync_target_frames', [10_000, 40_000]),
              "replay_start_size": 50000,
              "epsilon_decay_last_frame": epsilon_decay,
              "epsilon_start": 0.99,
              "epsilon_final": 0.000001,
              "amount_of_eval_rounds": 20 * 450,
              }

    time, iter_no, run_id = deep_q_main(params)

    run[f"trials/trials/{opt_id}/reward/time"].log(time)
    run[f"trials/trials/{opt_id}/reward/iter_no"].log(iter_no)
    return iter_no


n_amount = 10
study.optimize(objective, n_trials=n_amount, callbacks=[neptune_callback])
run.stop()


# commands = [[py_path, "/home/tyson/ReLe_final/Deep_Q-learning/deep_Q_learn_optuna.py", "-r", str(run_id), "-g", args.grid, "-n",
#              args.act] for i in range(n_amount)]
#
# for command in commands:
#     p = subprocess.Popen(command, stdout=subprocess.PIPE)
#     time.sleep(1)
# else:
#     (output, err) = p.communicate()
#     p_status = p.wait()
