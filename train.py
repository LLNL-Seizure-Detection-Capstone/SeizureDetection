import torch
import yaml
import matplotlib.pyplot as plt
import sys
import os

def load_yaml(path) :
    print('Reading yaml...')
    with open('train_config.yaml', "r") as yamlfile :
        config_data = yaml.safe_load(yamlfile)
        print("Read sucessful!!")
    return config_data

def load_data(config_data) :
    # TODO Implement functionality to take a dataset path and return train and test loaders in pytorch
    dataset_path = config_data['dataset_path']
    batch_size = config_data['batch_size']
    return 'TRAIN LOADER', 'TEST LOADER'

def load_model(config_data) :
    # TODO Intiliaze the value of the model
    model_type =  config_data['model']
    return 'Model'

def load_optimizer(model_params, config_data) :
    # Lowercase and remove white space to limit user input errors
    optimizer_type = config_data['optimizer'].lower().replace(' ', '')
    if optimizer_type == 'adam' :
        learning_rate = config_data['learning_rate']
        return torch.optim.Adam(model_params, learning_rate)
    else :
        print('ERROR: Optimizer Type is not recognized')

def load_loss_function(config_data) :
    # Lowercase and remove white space to limit user input errors
    loss_function_type = config_data['loss_function'].lower().replace(' ', '')
    if loss_function_type == 'Cross Entropy'.lower().replace(' ', '') :
        return torch.nn.CrossEntropyLoss()
    else :
        print('ERROR: Unknown Loss Function in config data')


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
    train_loader, test_loader = load_data(config_data)
    model = load_model(config_data)
    optimizer = load_optimizer(model.parameters(), config_data)
    loss_fn = load_loss_function(config_data)
    print(train_loader, test_loader, model, optimizer, loss_fn, EPOCHS)
    #train(train_loader, test_loader, model, optimizer, loss_fn, EPOCHS)





