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

feature_size = 512
sequence_size = 24
# 2 second chunk is size (512, 24)

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
train_dl, test_dl = load_train_data(config_data)

src = torch.as_strided(config_data,(sequence_size,feature_size),(1,1)).unsqueeze(1)
target = torch.as_strided(config_data[1:],(sequence_size,feature_size),(1,1)).unsqueeze(1)
print(target.size())
# size = (sequence, batch, features)
