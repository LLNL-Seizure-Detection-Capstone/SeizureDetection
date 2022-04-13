import torch
import matplotlib.pyplot as plt
import sys
import os
from models import *
from utils import *
import time
from tqdm import tqdm
from torchvision import transforms
import math
import torch.nn as nn
import torch.nn.functional as F
import random

# Mask 1 second sections of features
def mask_features(features):
    # loop through columns of df features
    for i in range(len(features)):
        prob = random.random() # generates value between 0 and 1
        if prob < 0.15:
            prob /= 0.15
            # 80% randomly change token to mask token
            if prob < 0.8:
                features[i] = torch.full(size=features[i].size(), fill_value=np.inf)
            # 10% randomly change token to random token
            elif prob < 0.9:
                features[i] = torch.rand(size=features[i].size())

    return features

feature_size = 24
sequence_size = 512
# 2 second chunk is size (512, 24)
nrow_to_drop = 1

if __name__ == "__main__" :
    # If no command line arguement were given then it checks the local directory for a train_config.yaml file
    if len(sys.argv) == 1 :
        config_path = sys.argv[0]
        config_path = config_path.split(os.sep)
        config_path[-1] = 'train_config.yaml'
        config_path = os.sep.join(config_path)
    # Uses the command line arguement as a train_config.yaml file
    elif len(sys.argv) == 2 :
        config_path = sys.argv[1]
    # Error if more that 3 arguements are passed into the project
    else :
        print('ERROR: Expected a 1 Arguement in train.py (yaml config file)')

config_data = load_yaml(config_path)
dataset_path = config_data['dataset_path']
chbmit_dataset = RawCHBMITDataset('./raw_chb01')
length_of_dataset = chbmit_dataset.__len__() 

features = torch.empty(size=(57599, 23, 256)) # 921600 = 60 * 60 * 256 
targets = torch.empty(size=(57599, 23, 256))
#TODO: Fill feature and target tensor with getitem()

#TODO: Build/train workable transformer
tf_model= nn.Transformer(feature_size,8,2,2,2,0.2) # build model
optimizer = torch.optim.SGD(tf_model.parameters(),lr=0.1)
binary_loss_fn = nn.BCELoss()
total_loss = 0
for epoch in range(100):
    # mask inputs
    src = mask_features(features) # mask features
    out = tf_model(src,targets)
    optimizer.zero_grad()
    out = torch.sigmoid(out) # run outputs through a sigmoid to be between (0, 1)
    loss = binary_loss_fn(out,targets)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()

    avg_train_loss = total_loss / len(out) 
    print("Average training loss:{0:.2f}".format(avg_train_loss))
out = tf_model(src,targets)
