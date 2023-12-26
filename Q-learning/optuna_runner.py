import time
import subprocess
import neptune
import os
import optuna
import argparse

parser = argparse.ArgumentParser(description='Q-learn start optimizers')
parser.add_argument("-g", "--grid", required=True, help="grid-size")
parser.add_argument("-n", "--act", required=True, help="number_of_actions")
args = parser.parse_args()

py_path = "/home/tyson/.conda/envs/rele/bin/python3"

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token
)  # your credentials

run["Algo"] = "Q_learning"
run["grid_size"] = args.grid
run["number_of_actions"] = args.act
run_id = run["sys/id"].fetch()

params = {"number_of_actions": int(args.act),
          "grid_size": int(args.grid),
          "epsilon": 0.2,
          "gamma": 0.9,
          "decay": "to_be_tuned",
          "test_episodes": 20,
          "amount_of_eval_rounds": 100}

run["parameters"] = params

optuna.delete_study(study_name=f"Q_learn_study_grid_{args.grid}_act_{args.act}", storage="sqlite:///example.db")
study = optuna.create_study(storage="sqlite:///example.db", study_name=f"Q_learn_study_grid_{args.grid}_act_{args.act}",
                            direction="minimize")
n_amount = 20

commands = [[py_path, "/home/tyson/ReLe_final/Q-learning/Q_learn_optuna.py", "-r", str(run_id), "-g", args.grid, "-n",
             args.act] for i in range(n_amount)]

for command in commands:
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    time.sleep(1)
else:
    (output, err) = p.communicate()
    p_status = p.wait()

# good old code:
#
# py_path = "/home/tyson/.conda/envs/lunar_lander/bin/python3"
#
# token = os.getenv('NEPTUNE_API_TOKEN')
# run = neptune.init_run(
#     project="ReL/ReLe-opt",
#     api_token=token
# )  # your credentials
#
# run_id = run["sys/id"].fetch()
#
# params = {"direction": "minimize", "n_trials": 20}
# run["parameters"] = params
#
# optuna.delete_study(study_name="Q_learn_study", storage="sqlite:///example.db")
# study = optuna.create_study(storage="sqlite:///example.db", study_name="Q_learn_study", direction="minimize")
# n_amount = 3
#
# commands = [[py_path, "/home/tyson/ReLe_final/Q-learning/opt_test.py", "-r", str(run_id)]
#             for i in range(n_amount)]
#
# for command in commands:
#     p = subprocess.Popen(command, stdout=subprocess.PIPE)
#     time.sleep(1)
# else:
#     (output, err) = p.communicate()
#     p_status = p.wait()