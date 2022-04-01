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

def accuracy(y_true, y_prob):
    y_hat = []
    num_correct = 0
    print(type(y_true))
    print(type(y_hat))

    # convert probabilities to binary classifier
    for y in y_prob:
        #print("y", y)
        if y >= 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)
    # compare y_true to y_hat
    for indx, y_value in enumerate(y_true):
        if y_value == y_hat[indx]:
            num_correct += 1
    
    return num_correct / len(y_true)

def mask_features(features):
    # tokens_len = [len(token) for token in tokens]
    # chars = [char for char in sentence]
    # output_label = []

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
nrow_to_drop = 1 # change this value to cut off more rows

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
chbmit_dataset = RawCHBMITDataset('./raw_chb01')

length_of_dataset = chbmit_dataset.__len__() 
print("length", length_of_dataset)
# drop out a whole 2-second chunk
features = torch.empty(size=(57599, 23, 256)) # 921600 = 60 * 60 * 256 
targets = torch.empty(size=(57599, 23, 256))

count = 0
for i in range(length_of_dataset):
    feature = chbmit_dataset.__getitem__(i)[0] # we don't actually need the target tho
    print("feature size: ", feature.shape)
    print("features[i] size: ", features[i].shape)
    features[i] = feature
    if i < length_of_dataset-1:
        target = chbmit_dataset.__getitem__(i+1)[0]
        targets[i] = target
    else:
        targets[i] = torch.full(size=(23, 256), fill_value=-1)

print("features:", features)
print("targets:", targets)

# TODO do we need these?
src = torch.as_strided(features,(sequence_size,feature_size),(1,1)).unsqueeze(1)
target = torch.as_strided(targets,(sequence_size,feature_size),(1,1)).unsqueeze(1) # target no longer contains any 1 values (tensor full of zeroes)

# size = (sequence, batch, features)
tf_model= nn.Transformer(feature_size,8,2,2,2,0.2)

# src_mask = tf_model.generate_square_subsequent_mask(sequence_size)
optimizer = torch.optim.SGD(tf_model.parameters(),lr=0.1)
loss_fn = torch.nn.MSELoss()
binary_loss_fn = nn.BCELoss()
# try using binary cross entropy

#train_acc = 0
total_loss = 0

for epoch in range(100):
    # mask inputs

    out = tf_model(src,target)
    optimizer.zero_grad()
    out = torch.sigmoid(out) # run outputs through a sigmoid to be between (0, 1)
    loss = binary_loss_fn(out,target)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    #curr_epoch_acc = flat_accuracy()
    #train_acc += curr_epoch_acc

    avg_train_loss = total_loss / len(out) 
    print("Average training loss:{0:.2f}".format(avg_train_loss))

out = tf_model(src,target)
