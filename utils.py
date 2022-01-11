#==========================================================================================================#
# UTILS.PY
# This file is meant to hold all auxilary functions that aid in data prepartion for training or predicting
#==========================================================================================================#

import yaml
from models import CNN_AE_MLP

def load_yaml(path) :
    print('Reading yaml...')
    with open('train_config.yaml', "r") as yamlfile :
        config_data = yaml.safe_load(yamlfile)
        print("Read sucessful!!")
    return config_data

def load_train_data(config_data) :
    # TODO Implement functionality to take a dataset path and return train and test loaders in pytorch
    dataset_path = config_data['dataset_path']
    batch_size = config_data['batch_size']
    return 'TRAIN LOADER', 'TEST LOADER'

def load_new_model(config_data) :
    model_type =  config_data['model'].lower().replace(' ', '')
    if model_type == 'cnn_ae_mlp' :
        return CNN_AE_MLP()
    else :
        print('Model not found')
        return 'ERROR in load_new_model'

def load_existing_model(config_data) :
    model_path = config_data['model_load_path']
    model = torch.load(model_path)
    model.eval()
    return model

def load_optimizer(model_params, config_data) :
    # Lowercase and remove white space to limit user input errors
    optimizer_type = config_data['optimizer'].lower().replace(' ', '')
    if optimizer_type == 'adam' :
        learning_rate = config_data['learning_rate']
        return torch.optim.Adam(model_params, learning_rate)
    else :
        print('ERROR: Optimizer Type is not recognized')

def load_predict_data(config_data) :
    # TODO Implement functionality to get a single dataset to run predicitons on
    pass


