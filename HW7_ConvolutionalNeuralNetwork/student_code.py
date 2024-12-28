# Regular python imports
import os
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.c1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0, bias = True)
        self.mp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.c2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0, bias = True)
        self.mp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.l1 = nn.Linear(400, 256, bias = True)
        self.l2 = nn.Linear(256, 128, bias = True)
        self.l3 = nn.Linear(128, num_classes, bias=True)

    def forward(self, x):
        shape_dict = {}
        # certain operations
        conv1 = self.c1(x)
        x = nn.functional.relu(conv1)
        x = self.mp1(x)
        shape_dict.update({1:x.size()})

        conv2 = self.c2(x)
        x = nn.functional.relu(conv2)
        x = self.mp1(x)
        shape_dict.update({2:x.size()})

        x = x.view(-1, 400)
        shape_dict.update({3:x.size()})

        lin1 = self.l1(x)
        x = nn.functional.relu(lin1)
        shape_dict.update({4:x.size()})

        lin2 = self.l2(x)
        x = nn.functional.relu(lin2)
        shape_dict.update({5:x.size()})

        lin3 = self.l3(x)
        shape_dict.update({6:x.size()})

        return lin3, shape_dict

# Return the number of trainable parameters of LeNet.
def count_model_params():
    model = LeNet()
    model_params = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_params = model_params + param.nume1()
    model_params = model_params/(1e6)
    return model_params

# Model (torch.nn.module): The model created to train
# Train_loader (pytorch data loader): Training data loader
# Optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
# Criterion (nn.CrossEntropyLoss) : Loss function used to train the network
# Epoch (int): Current epoch number
def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
