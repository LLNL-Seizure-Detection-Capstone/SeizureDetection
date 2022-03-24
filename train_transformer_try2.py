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

length_of_dataset = chbmit_dataset.__len__() # 57600
print("length", length_of_dataset)
# drop out a whole 2-second chunk
features = torch.empty(size=(23, 921600)) # set to size of min num columns
targets = torch.empty(size=(23, 921600))

# TODO generate mask

# TODO ask Sara to repurpose dataloader for raw dataset
count = 0
for i in range(length_of_dataset):
    feature = chbmit_dataset.__getitem__(i)[0] # we don't actually need the target tho
    features[i] = feature
    if i < length_of_dataset-1:
        target = chbmit_dataset.__getitem__(i+1)[0]
        targets[i] = target
    else:
        targets[i] = torch.full(size=(512, 24), fill_value=-1)

# mask language model: ex. bert
#   use smaller architecture of bert (if available)-- but not the pretrained version
#   don't use distilbert
#   probably need supercomputer
#   pytorch has a transformer we can set up to be like bert
#       reduce number of heads

print("features:", features)
print("targets:", targets)

src = torch.as_strided(features,(sequence_size,feature_size),(1,1)).unsqueeze(1)

target = torch.as_strided(targets,(sequence_size,feature_size),(1,1)).unsqueeze(1) # target no longer contains any 1 values (tensor full of zeroes)
# as strided seems necessary because we want target to be of size (T, E) where T is target sequence length and E is the feature number
# however when we covert our list of 8190 targets we get a 512 x 1 x 24 tensor full of zeroes

# take output and push it through a neural net
# instead of the output be the next classification it could be the next 2 seconds (you're learning the pattern) & now you have a pretrained transformer.
# train the model by masking one of the 24 channels and having it predict the signal. Then you can retrain the whole thing and have it be a classifier

print("TARGET AFTER AS_STRIDED:\n", target)
print("Any 1's:", torch.any(target))
print("SUM TARGET:", torch.sum(target))
print("SUM SOURCE:", torch.sum(src))


# size = (sequence, batch, features)
tf_model= nn.Transformer(feature_size,8,2,2,2,0.2)

src_mask = tf_model.generate_square_subsequent_mask(sequence_size)
optimizer = torch.optim.SGD(tf_model.parameters(),lr=0.1)
loss_fn = torch.nn.MSELoss()
binary_loss_fn = nn.BCELoss()
# try using binary cross entropy

#train_acc = 0
total_loss = 0

for epoch in range(100):
    out = tf_model(src,target,src_mask)
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
# print("output shape:", out.size())
# sig_out = torch.sigmoid(out) # check out various thresholds (precision and recall) (24 x 1 x 512)
# print("sigmoid output shape:", sig_out.size())
# # print(target)
# print("OUPUT VALUES:\n", sig_out)
# #print("Accuracy: ", accuracy(targets, sig_out))

