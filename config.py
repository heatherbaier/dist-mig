import argparse


arg_lists = []
parser = argparse.ArgumentParser(description = "a3cRAM")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


training_args = add_argument_group("Training Arguments")
training_args.add_argument("--batch_size", 
                           type = int, 
                           default = 20, 
                           help = "batch size is equivalent to number of process to run asychronously / nodes * ppn")
training_args.add_argument("--tv_split", 
                           type = float, 
                           default = 1, 
                           help = "Train/Val split percentage - given as a float i.e. .75")
training_args.add_argument("--n_epochs", 
                           type = int, 
                           default = 1, 
                           help = "Number of epochs")
training_args.add_argument("--n_mini_epochs", 
                           type = int, 
                           default = 2, 
                           help = "Number of mini epochs performed on each image within an epoch")
training_args.add_argument("--eps_decay", 
                           type = int, 
                           default = 150, 
                           help = "Value by which to decay epsilon")


env_args = add_argument_group("RL Environment Arguments")
env_args.add_argument("--display", 
                      type = str, 
                      default = "False", 
                      help = "Whether to display the interactive RL environment during training - Cannot display on HPC")
env_args.add_argument("--n_actions", 
                      type = int, 
                      default = 5,
                      help = "How many actions the RL agent can choose from / output size of linear layer")


data_args = add_argument_group("Data Arguments")
data_args.add_argument("--imagery_dir",
                      type = str,
                      default = "/sciclone/geograd/Heather/mex_imagery/",
                      help = "Full path to directory containing imagery")
data_args.add_argument("--json_path",
                      type = str,
                      default = "/sciclone/home20/hmbaier/claw/migration_data.json",
                      help = "Full path to json containing muni_id -> num_migrants mapping.")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
