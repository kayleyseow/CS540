import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

"""
INPUT: 
    An optional boolean argument (default value is True for training dataset)

RETURNS:
     Dataloader for the training set (if training = True) or the test set (if training = False)
"""
def get_data_loader(training = True):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    train_data_set = datasets.FashionMNIST('./data',train=training, download=True,transform=transform)
    loader = torch.utils.data.DataLoader(train_data_set, batch_size = 64)
    return loader

"""
INPUT: 
    None

RETURNS:
    An untrained neural network model
"""
def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 ** 2, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

"""
INPUT: 
    model - the model produced by the previous function
    train_loader  - the train DataLoader produced by the first function
    criterion   - cross-entropy 
    T - number of epochs for training

RETURNS:
    None
"""
def train_model(model, train_loader, criterion, T):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        for iter, data in enumerate(train_loader, 0):
            inputs, labels = data
            # zero all of the param gradients before start
            optimizer.zero_grad()
            # the good stuff of the function
            outputs = model(inputs)
            _,pred_label = torch.max(outputs, dim = 1)
            correct  = correct + torch.sum(pred_label == labels)
            loss = criterion(outputs, labels)
            running_loss = running_loss + loss.item()
            loss.backward()
            optimizer.step()
        # Print the stats
        print(f'Train Epoch: {epoch}\tAccuracy: {correct:6d}/60000({(correct*100)/len(train_loader.dataset):.2f}%)\tLoss: {(running_loss/len(train_loader.dataset))*(train_loader.batch_size):.3f}')
        # Reset variables
        running_loss = 0.0
        correct = 0 
    
"""
INPUT: 
    model - the the trained model produced by the previous function
    test_loader    - the test DataLoader
    criterion   - cropy-entropy 

RETURNS:
    None
"""
def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct = correct + (predicted == labels).sum().item()
            loss = loss + criterion(outputs, labels)
    length = len(test_loader.dataset)
    if show_loss: 
        print(f'Average loss: {(loss/length)*(test_loader.batch_size):.3f}')
    print(f'Accuracy: {(correct*100)/length}%')

"""
INPUT: 
    model - the trained model
    test_images   -  test image set of shape Nx1x28x28
    index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

RETURNS:
    None
"""
def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    image = test_images[index]
    logits = model(image)
    prob = F.softmax(logits, 1)
    top_p, top_class = prob.topk(3, 1)
    top_p = top_p.tolist()[0]
    top_class = top_class.tolist()[0]
    for x in range(len(top_p)):
        print(f'{class_names[top_class[x]]}: {top_p[x] * 100:.2f}%')

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    # Test get_data_loader()
    '''
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    print(type(test_loader))
    print(test_loader.dataset)
    '''
    
    # Test build_model()
    '''
    model = build_model()
    print(model)
    '''
    
    # Test train_model(), evaluate_model()
    '''
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion)
    '''
    
    # Test predict_label()
    '''
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    pred_set = images
    predict_label(model, pred_set, 1)
    '''
