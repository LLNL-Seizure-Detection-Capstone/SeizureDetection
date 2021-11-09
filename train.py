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

def accuracy(logits, labels) :
    correct = 0
    pred = logits.max(dim=1)[1]
    correct += pred.eq(labels).sum().item()
    return correct / len(labels)

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

def train(train_loader, test_loader, model, optimizer, loss_fn, epochs) :
    train_accs, test_accs, train_losses, test_losses = list(), list(), list(), list()
    print('Starting Training...')
    for epoch in range(epochs) :
        T0 = time.time()
        epoch_train_accs, epoch_test_accs, epoch_train_losses, epoch_test_losses = list(), list(), list(), list()
        for batch in train_loader :
            data = batch[0]
            labels = batch[1]
            model.train()
            optimizer.zero_grad()
            out = model(data)
            train_loss = loss_fn(out, labels)
            train_loss.backward()
            optimizer.step()
            train_acc = accuracy(out, labels)
            epoch_train_accs.append(train_acc)
            epoch_train_losses.append(train_loss.item())

        for batch in test_loader :
            model.eval()
            data = batch[0]
            labels = batch[1]
            optimizer.zero_grad()
            out = model(data)
            test_loss = loss_fn(out, labels)
            test_acc = accuracy(out, labels)
            epoch_test_accs.append(test_acc)
            epoch_test_losses.append(test_loss.item())

        train_acc = sum(epoch_train_accs) / len(epoch_train_accs)
        train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        test_acc = sum(epoch_test_accs) / len(epoch_test_accs)
        test_loss = sum(epoch_test_losses) / len(epoch_test_losses)

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
            

        print('Epoch: {:03d}, Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, Time: {}s'.format(epoch, train_loss, train_acc,  test_loss, test_acc, round(time.time() - T0)))
    print('Training Finished...')
    display_plot(train_accs, test_accs, train_losses, test_losses)

def save_model(model, config_data) :
    save_path = config_data['model_save_path']
    if save_path.lower.replace(' ', '') == 'none' :
        print('No Save Path Specified')
        return
    print('Saving Model...')
    torch.save(model, save_path)
    print('Model Saved')

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
    print(config_data)

    epochs = config_data['epochs']
    train_loader, test_loader = load_train_data(config_data)
    model = load_new_model(config_data)
    optimizer = load_optimizer(model.parameters(), config_data)
    loss_fn = load_loss_function(config_data)
    #train(train_loader, test_loader, model, optimizer, loss_fn, epochs)
    save_model(model, config_data)

   



