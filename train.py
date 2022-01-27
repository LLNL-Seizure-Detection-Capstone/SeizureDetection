#==========================================================================================================#
# TRAIN.PY
# This file is meant to hold the training structure updating model parameters.
#==========================================================================================================#
import torch
import matplotlib.pyplot as plt
import sys
import os
from models import *
from utils import *
import time
from tqdm import tqdm



def accuracy(logits, labels) :
    predicitons = torch.tensor( [ 1 if x > .5 else 0 for x in logits ] )
    correct = torch.sum(predicitons == labels)
    return (correct / len(labels)).item()

def display_plot(train_accs, test_accs, train_losses, test_losses):
    plt.title('Accuracy Graph')
    plt.plot(test_accs, label="Test Acc")
    plt.plot(train_accs, label="Train Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc= 'lower right')
    plt.grid(axis = 'y')
    plt.show()

    plt.title('Loss Graph')
    plt.plot(train_losses, label= 'Train Loss')
    plt.plot(test_losses, label= 'Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc= 'lower right')
    plt.grid(axis = 'y')
    plt.show()

def save_model(model, config_data) :
    save_path = config_data['model_save_path']
    if save_path.lower.replace(' ', '') == 'none' :
        print('No Save Path Specified')
        return
    print('Saving Model...')
    torch.save(model, save_path)
    print('Model Saved')

def get_train_loop(config_data) :
    model_type =  config_data['model'].lower().replace(' ', '')
    if model_type == 'cnn_ae_mlp' :
        return train_loop_CNN_AE_MLP
    else :
        print('Training loop was not found')
        return 'ERROR in get_train_loop'

def train_loop_CNN_AE_MLP(train_loader, test_loader, model, optimizer, epochs) :
    print('Starting Training...')
    ae_loss_fn = torch.nn.MSELoss()
    target_loss_fn = torch.nn.BCELoss()
    for epoch in range(epochs) : 
        T0 = time.time()
        epoch_acc = list()
        for batch in tqdm(train_loader) :
            data = batch[0].float()
            labels = batch[1].float()
            model.train()
            optimizer.zero_grad()
            decoded_mat, out = model(data)
            out = out.flatten()
            ae_train_loss = ae_loss_fn(decoded_mat, data)
            target_train_loss = target_loss_fn(out, labels)
            ae_train_loss.backward(retain_graph=True)
            target_train_loss.backward(retain_graph=True)
            optimizer.step()
            train_acc = accuracy(out, labels)

        for batch in tqdm(test_loader) :
            model.eval()
            optimizer.zero_grad()
            data = batch[0].float()
            labels = batch[1].float()
            decoded_mat, out = model(data)
            out = out.flatten()
            ae_test_loss = ae_loss_fn(decoded_mat, data)
            target_test_loss = target_loss_fn(out, labels)
            test_acc = accuracy(out, labels)
            epoch_acc.append(test_acc)

        print('Test Acc after one Epoch: {}'.format((sum(epoch_acc) / len(epoch_acc))))

    return model

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
    epochs = config_data['epochs']
    train_loader, test_loader = load_train_data(config_data)
    tensor = next(iter(train_loader))
    model = load_new_model(config_data)
    optimizer = load_optimizer(model.parameters(), config_data)
    train_loop = get_train_loop(config_data)
    model = train_loop_CNN_AE_MLP(train_loader, test_loader, model, optimizer, epochs)
    save_model(model, config_data)

   



