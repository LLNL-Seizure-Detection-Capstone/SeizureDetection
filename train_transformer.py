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
# train_dl, test_dl = load_train_data_(config_data)

dataset_path = config_data['dataset_path']
chbmit_dataset = CHBMITDataset(dataset_path)

length_of_dataset = chbmit_dataset.__len__() # 8190

features = torch.empty(size=(8190, 512, 24)) # we want to initialize empty tensors (512 x 24 x 8190)
targets = torch.empty(length_of_dataset) # (1 x 8190)

for i in range(length_of_dataset):
    feature, target = chbmit_dataset.__getitem__(i)
    features[i] = feature
    targets[i] = target


src = torch.as_strided(features,(sequence_size,feature_size),(1,1)).unsqueeze(1)
target = torch.as_strided(targets,(sequence_size,feature_size),(1,1)).unsqueeze(1)

# size = (sequence, batch, features)
tf_model= nn.Transformer(feature_size,8,2,2,2,0.2)

src_mask = tf_model.generate_square_subsequent_mask(sequence_size)
optimizer = torch.optim.SGD(tf_model.parameters(),lr=0.1)
loss_fn = torch.nn.MSELoss()
binary_loss_fn = nn.BCELoss()
# try using binary cross entropy

for epoch in range(1000):
    out = tf_model(src,target,src_mask)
    optimizer.zero_grad()
    out = torch.sigmoid(out) # run outputs through a sigmoid to be between (0, 1)
    loss = binary_loss_fn(out,target)
    loss.backward()
    optimizer.step()

out = tf_model(src,target)
sig_out = torch.sigmoid(out) # check out various thresholds (precision and recall)
print(target)
print(sig_out)

