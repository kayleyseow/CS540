
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import itertools

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=transform) 
 
    test_set = datasets.FashionMNIST('./data', train=False,transform=transform)
    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)

    return loader

train_loader = get_data_loader()
print(type(train_loader))
print(train_loader.dataset)
test_loader = get_data_loader(False)



def build_model():
    model = nn.Sequential( 
        nn.Flatten(),
        nn.Linear(28**2, 128), 
        nn.ReLU(), 
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )
    return model

model = build_model()
print(model)


def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):  # loop over the dataset multiple times

        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item() * labels.size(0)
        print(f'Training Epoch: {epoch}    Accuracy: {correct}/{total}({100*correct/total:.2f}%) Loss:{running_loss/total:.3f}')
        running_loss = 0.0

train_model(model, train_loader, nn.CrossEntropyLoss(), 5)  


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
... (47 lines left)
Collapse
message.txt
5 KB
﻿
from urllib.request import proxy_bypass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import itertools

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=transform) 
 
    test_set = datasets.FashionMNIST('./data', train=False,transform=transform)
    if training:
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64)

    return loader

train_loader = get_data_loader()
print(type(train_loader))
print(train_loader.dataset)
test_loader = get_data_loader(False)



def build_model():
    model = nn.Sequential( 
        nn.Flatten(),
        nn.Linear(28**2, 128), 
        nn.ReLU(), 
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )
    return model

model = build_model()
print(model)


def train_model(model, train_loader, criterion, T):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):  # loop over the dataset multiple times

        correct = 0
        total = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item() * labels.size(0)
        print(f'Training Epoch: {epoch}    Accuracy: {correct}/{total}({100*correct/total:.2f}%) Loss:{running_loss/total:.3f}')
        running_loss = 0.0

train_model(model, train_loader, nn.CrossEntropyLoss(), 5)  


def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item() * labels.size(0)
        
        if show_loss:
            print(f'Average loss: {running_loss/total:.4f}')
        print(f'Accuracy: {100*correct/total:.2f}%')
        running_loss = 0.0

evaluate_model(model, test_loader, nn.CrossEntropyLoss(), False)
evaluate_model(model, test_loader, nn.CrossEntropyLoss())


def predict_label(model, test_images, index):
    with torch.no_grad():
        prediction = model(test_images[index])

        probs = F.softmax(prediction, dim=1)
        probs = probs.flatten().tolist()

        percentages = [prob * 100 for prob in probs]
    
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal',
                    'Shirt' ,'Sneaker','Bag','Ankle Boot']

    final_dict = dict(zip(class_names, percentages))
    sorted_dict = {key: val for key, val in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)}
    
    top3 = dict(itertools.islice(sorted_dict.items(), 3))

    for key in top3:
        print(f'{key}: {top3[key]:.2f}%')

dataiter = iter(test_loader)
images, labels = dataiter.next()
predict_label(model, images, 1)


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    
message.txt
5 KB