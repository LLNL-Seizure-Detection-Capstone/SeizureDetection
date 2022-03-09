#==========================================================================================================#
# UTILS.PY
# This file is meant to hold all auxilary functions that aid in data prepartion for training or predicting
#==========================================================================================================#

import yaml
from models import CNN_AE_MLP
import os
import pandas as pd
import torch
from datasets import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_yaml(path) :
    print('Reading yaml...')
    with open('train_config.yaml', "r") as yamlfile :
        config_data = yaml.safe_load(yamlfile)
        print("Read sucessful!!")
    return config_data

def load_train_data(config_data) :
    dataset_path = config_data['dataset_path']
    batch_size = config_data['batch_size']
    if 'chbmit' in dataset_path.lower() :
        chbmit_dataset = CHBMITDataset(dataset_path)
        test_size = len(chbmit_dataset) // 4
        train_size = len(chbmit_dataset) - test_size
        train_dataset, test_dataset = torch.utils.data.random_split(chbmit_dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size), shuffle=True)
        return train_loader, test_loader

    else :
        print('ERROR in load_train_data')
        return 'TRAIN LOADER', 'TEST LOADER' 

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

def k_fold_split(config_data) :
    k = config_data['k_size']
    dataset_path = config_data['dataset_path']
    batch_size = config_data['batch_size']
    if 'chbmit' in dataset_path.lower() :
        chbmit_dataset = CHBMITDataset(dataset_path)
        split_size = len(chbmit_dataset) // k
        splits = [ split_size for x in range(k) ]
        splits[-1] = splits[-1] + len(chbmit_dataset) - sum(splits)
        datasets = torch.utils.data.random_split(chbmit_dataset, splits)
        loaders = list()
        for i in range(k) :
            train_datasets = list()
            test_dataset = list()
            if i != 0 :
                train_datasets = train_datasets + datasets[:i]
            if i != len(datasets) - 1 :
                train_datasets = train_datasets + datasets[i+1:]
            test_dataset = datasets[i]
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size), shuffle=True)
            loaders.append( (train_loader, test_loader) )
        return loaders

    else :
        print('ERROR in load_train_data')
        return 'TRAIN LOADER', 'TEST LOADER' 

def load_predict_data(config_data) :
    # TODO Implement functionality to get a single dataset to run predicitons on
    pass

def test_class_balance(config_data) :
        all_balances = list()
        test_iters = 50
        for x in tqdm(range(test_iters)) :
            loaders = k_fold_split(config_data)
            k = len(loaders)
            for loader in loaders :
                tester = loader[1]
                vals = [  sum(batch[1]).item() /  len(batch[1]) for batch in tester ]
                class_balance = sum(vals) / len(vals)
                all_balances.append(class_balance)
        plt.boxplot(all_balances)
        plt.title(f'Class Balance over {test_iters*k} Samples')
        plt.show()

def get_train_loop(config_data) :
    model_name = config_data['model']
    if model_name == 'CNN_AE_MLP' :
        return train_loop_CNN_AE_MLP
    else :
        print('ERROR: Could not find the correct training loop')
        return



class SeizureDetectionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df.insert(loc=23, column='padding', value=0) # add a column of padding to dataframe
        
        print( "seizure starts ", self.df.iloc[2996]) # this row should have an outcome of 1 but instead is printing outcome 0?
        # at [2996] seconds the seizure starts so this should be (512 * 2996) + 2996 = 1,536,948 <- this row should have an outcome of 1 but it doesn't exist
        #print(self.df.head(2))

        #print(self.df.iloc[4:8, 24])
        # self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df) # number of rows in csv file (2 second chunks)

    def __getitem__(self, index):
        startRow = (512 * index) + index # I am not sure if this ignores header or not?
        endRow = startRow + 512

        outcome = 1 if 1.0 in self.df.iloc[startRow:endRow, 24] else 0 # if there is at least a single 1 in any of the rows return 1 else 0

        return self.df.iloc[startRow:endRow, 0:24], outcome
        # in reality this needs to return 2 values and the second will prob be majority output of every row we selected for this index
        
        # locate time stamp
        # read in that edf file
        # set y_label = (0 or 1 depending on seizure)

        # if self.transform:
        #     file = self.transform(file)
        # return (file, y_label)

        #return a dataframe 512 x 24 with the last column padded with zeroes and the y label
