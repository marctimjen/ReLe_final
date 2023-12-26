import time
import subprocess
import uuid

import neptune
import os
import optuna

py_path = "/home/tyson/.conda/envs/lunar_lander/bin/python3"

token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="ReL/ReLe-opt",
    api_token=token
)  # your credentials

run_id = run["sys/id"].fetch()

params = {"direction": "minimize", "n_trials": 20}
run["parameters"] = params

optuna.delete_study(study_name="Q_learn_study", storage="sqlite:///example.db")
study = optuna.create_study(storage="sqlite:///example.db", study_name="Q_learn_study", direction="minimize")
n_amount = 3

commands = [[py_path, "/home/tyson/ReLe_final/Q-learning/opt_test.py", "-r", str(run_id), "-i", str(i)]
            for i in range(n_amount)]

for command in commands:
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    time.sleep(1)
else:
    (output, err) = p.communicate()
    p_status = p.wait()
