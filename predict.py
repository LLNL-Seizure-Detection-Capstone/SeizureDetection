#==========================================================================================================#
# PREDICT.PY
# This file is holds the functionality to predict data given a path to a specific model.
#==========================================================================================================#

from models import *
from utils import *
import sys

if __name__ == "__main__" :
    # If no command line arguement were given then it checks the local directory for a predict_config.yaml file
    if len(sys.argv) == 1 :
        config_path = sys.argv[0]
        config_path = config_path.split(os.sep)
        config_path[-1] = 'predict_config.yaml'
        config_path = os.sep.join(config_path)
    # Uses the command line arguement as a train_config.yaml file
    elif len(sys.argv) == 2 :
        config_path = sys.argv[1]
    # Error if more that 3 arguements are passed into the project
    else :
        print('ERROR: Expected a 1 Arguement in train.py (yaml config file)')

    config_data = load_yaml(config_path)
    model = load_existing_model(config_data)
    dataset = load_predict_data(config_data)
    predicitons = model(dataset)
    save_predictions(dataset, predictions, config_data)
