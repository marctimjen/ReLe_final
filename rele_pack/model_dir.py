import sys
import os
import torch

def model_dir(script_directory):
    """
    This function creates a directory for the model to be saved in.

    :return: path of dir where model is to be saved
    """

    script_directory = script_directory + "/model_saves/"

    if os.path.exists(script_directory):
        dir_list = os.listdir(script_directory)
        max_nr = max(int(vals.split("_")[1]) for vals in dir_list)
        path = script_directory + "model_" + str(max_nr + 1) + "/"
        os.makedirs(path, exist_ok=False)
    else:
        path = script_directory + "model_0/"
        os.makedirs(path, exist_ok=False)

    return path


def model_saver(model, save_path: str, game_idx: int, reward: float, final: bool = False):
    if final:
        torch.save(model.state_dict(), save_path + f"model_idx_{game_idx}_rew_{round(reward, 3)}_final.dat")
    else:
        torch.save(model.state_dict(), save_path + f"model_idx_{game_idx}_rew_{round(reward, 3)}.dat")

def model_path_loader(model_number: int = -1):
    script_directory = os.path.dirname(sys.argv[0])
    script_directory = script_directory + "/model_saves/"

    if os.path.exists(script_directory):
        dir_list = os.listdir(script_directory)
        max_nr = max(int(vals.split("_")[1]) for vals in dir_list)
    else:
        raise ValueError("No model directory found")

    if model_number == -1:
        load_path = script_directory + "model_" + str(max_nr) + "/"
    else:
        load_path = script_directory + "model_" + str(model_number) + "/"
        if not(os.path.exists(load_path)):
            raise ValueError("No model directory found with model number " + str(model_number))

    return load_path

def model_loader(path: str, print_models: bool = False):
    if print_models:
        dir_list = os.listdir(path)
        print("available_models:", dir_list)
    else:
        params = torch.load(path)
        return params