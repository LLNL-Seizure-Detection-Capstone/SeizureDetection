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

def display_plot(config_data, train_accs, test_accs, train_losses, test_losses, ae_train_losses=None, ae_test_losses=None):
    filepath = config_data['graph_save_path']

    plt.title('Accuracy Graph')
    plt.plot(test_accs, label="Test Acc")
    plt.plot(train_accs, label="Train Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc= 'lower right')
    plt.grid(axis = 'y')
    if filepath.lower() != 'none' :
        plt.savefig(filepath+'Accuracy.png')
    else :
        plt.show()

    plt.clf()

    plt.title('Target Loss Graph')
    plt.plot(train_losses, label= 'Train Loss')
    plt.plot(test_losses, label= 'Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc= 'lower right')
    plt.grid(axis = 'y')
    if filepath.lower() != 'none' :
        plt.savefig(filepath+'TargetLoss.png')
    else :
        plt.show()

    plt.clf()

    if ae_train_losses is not None and ae_test_losses is not None:
        plt.title('AutoEncoder Loss Graph')
        plt.plot(ae_train_losses, label= 'Train Loss')
        plt.plot(ae_test_losses, label= 'Test Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc= 'lower right')
        plt.grid(axis = 'y')
        if filepath.lower() != 'none' :
            plt.savefig(filepath+'AELoss.png')
        else :
            plt.show()

        plt.clf()

def save_model(model, config_data) :
    save_path = config_data['model_save_path']
    if save_path.lower().replace(' ', '') == 'none' :
        print('No Save Path Specified')
        return
    print('Saving Model...')
    torch.save(model.state_dict(), save_path)
    print('Model Saved')

def get_train_loop(config_data) :
    model_type =  config_data['model'].lower().replace(' ', '')
    if model_type == 'cnn_ae_mlp' :
        return train_loop_CNN_AE_MLP
    else :
        print('Training loop was not found')
        return 'ERROR in get_train_loop'

def train_loop_CNN_AE_MLP(train_loader, test_loader, model, optimizer, config_data) :
    print('Starting Training...')
    epochs = config_data['epochs']
    ae_loss_fn = torch.nn.MSELoss()
    target_loss_fn = torch.nn.BCELoss()
    
    total_train_acc = list()
    total_train_ae_loss = list()
    total_train_target_loss = list()

    total_test_acc = list()
    total_test_ae_loss = list()
    total_test_target_loss = list()

    for epoch in range(epochs) : 
        T0 = time.time()

        epoch_train_acc = list()
        epoch_train_ae_loss = list()
        epoch_train_target_loss = list()

        epoch_test_acc = list()
        epoch_test_ae_loss = list()
        epoch_test_target_loss = list()

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

            epoch_train_acc.append(train_acc)
            epoch_train_ae_loss.append(ae_train_loss.item())
            epoch_train_target_loss.append(target_train_loss.item())

        avg_train_acc =  round((sum(epoch_train_acc) / len(epoch_train_acc)), 4)
        avg_train_ae_loss = round((sum(epoch_train_ae_loss) / len(epoch_train_ae_loss)), 4)
        avg_train_target_loss = round((sum(epoch_train_target_loss) / len(epoch_train_target_loss)), 4)

        total_train_acc.append(avg_train_acc)
        total_train_ae_loss.append(avg_train_ae_loss)
        total_train_target_loss.append(avg_train_target_loss)

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

            epoch_test_acc.append(test_acc)
            epoch_test_ae_loss.append(ae_test_loss.item())
            epoch_test_target_loss.append(target_test_loss.item())

        avg_test_acc =  round((sum(epoch_test_acc) / len(epoch_test_acc)), 4)
        avg_test_ae_loss = round((sum(epoch_test_ae_loss) / len(epoch_test_ae_loss)), 4)
        avg_test_target_loss = round((sum(epoch_test_target_loss) / len(epoch_test_target_loss)), 4)

        total_test_acc.append(avg_test_acc)
        total_test_ae_loss.append(avg_test_ae_loss)
        total_test_target_loss.append(avg_test_target_loss)

        print('Epoch: {} \nTrain Acc: {}% Train AE Loss: {} Train Target Loss: {} \nTest Acc: {}% Test AE Loss: {} Test Target Loss:{}\n'\
            .format(epoch+1, avg_train_acc*100, avg_train_ae_loss, avg_train_target_loss, avg_test_acc*100, avg_test_ae_loss, avg_test_target_loss))

    display_plot(config_data, total_train_acc, total_test_acc, total_train_target_loss, total_test_target_loss, ae_train_losses=total_train_ae_loss, ae_test_losses=total_test_ae_loss)
    
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
    train_loader, test_loader = load_train_data(config_data)
    tensor = next(iter(train_loader))
    model = load_new_model(config_data)
    optimizer = load_optimizer(model.parameters(), config_data)
    train_loop = get_train_loop(config_data)
    model = train_loop_CNN_AE_MLP(train_loader, test_loader, model, optimizer, config_data)
    save_model(model, config_data)

   



