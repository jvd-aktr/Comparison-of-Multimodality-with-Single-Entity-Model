# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:10:21 2023

@author: akhtarj
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



## loading the dataset from device.

# Define the transforms for data augmentation and normalization
data_transforms = transforms.Compose([transforms.ToTensor()])


# Load the dataframe
path_pos = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\mnist_likedata\posImgPixel_data.csv'
path_neg = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\mnist_likedata\negImgPixel_data.csv'

df_pos = pd.read_csv(path_pos)
df_neg = pd.read_csv(path_neg)

# concatenate the dataframes
df = pd.concat([df_pos, df_neg])

# shuffle the instances
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna()
print(df)

# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Implement your data loading logic here
        # For example, if your dataframe has columns "image_path" and "label":
        # image = load_image(self.data.loc[idx, 'image_path'])
        # label = self.data.loc[idx, 'label']
        # return image, label
        pass

# Create the dataset
dataset = MyDataset(df)

# Split the dataset into train, test, and validation sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create the data loaders for each set
batch_size = 64
mnist_train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
mnist_valid_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
mnist_test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ## Step 1: loading and preprocessing MNIST dataset
# transform = transforms.Compose([transforms.ToTensor()])


# image_path = ''  # just to provide a value to 'root'
# transform = transforms.Compose([transforms.ToTensor()])
# mnist_dataset = torchvision.datasets.MNIST(root=image_path, train=True,transform=transform, download=True)
# mnist_valid_dataset = torch.utils.data.Subset(mnist_dataset,torch.arange(10000))
# mnist_train_dataset = torch.utils.data.Subset(mnist_dataset,torch.arange(10000, len(mnist_dataset)))
# mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, train=False,transform=transform, download=False)
# print('number of items in mnist_dataset:', len(mnist_dataset))
print('number of items in mnist_train_dataset:', len(mnist_train_dataset))
print('number of items in mnist_valid_dataset:', len(mnist_valid_dataset))
print('number of items in mnist_test_dataset:', len(mnist_test_dataset))

## Construct the data loader
batch_size = 64
torch.manual_seed(1)

train_dl = mnist_train_dataset
valid_dl = mnist_valid_dataset

## Construct a CNN in PyTorch
model = nn.Sequential()

# tune the model by changing the parameters
model.add_module('conv1',nn.Conv2d(in_channels=1, out_channels=4,kernel_size=3,stride=1, padding=0))
model.add_module('relu1',nn.ReLU())
model.add_module('pool1',nn.MaxPool2d(kernel_size=2, stride=2))

model.add_module('conv2',nn.Conv2d(in_channels=4,out_channels=2,kernel_size=3, stride=3, padding=0))
model.add_module('relu2',nn.ReLU())
model.add_module('pool2',nn.MaxPool2d(kernel_size=4, stride=4))

x = torch.ones((4, 1, 64, 64))
model(x).shape

model.add_module('flatten', nn.Flatten())
x = torch.ones((4, 1, 64, 64))
model(x).shape

model.add_module('fc1', nn.Linear(2,1024))  # to match the best condition mentioned in report, change 2 to 3136
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))
model.add_module('fc2', nn.Linear(1024, 10))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# define a training model
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl: 
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0) 
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float() 
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        print(f'Epoch {epoch+1}:: train_accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

torch.manual_seed(1)
num_epochs = 1
device = 'cpu'
hist = train(model, num_epochs, train_dl, valid_dl)

'''
## Plot loss function and accuracy
x_arr = np.arange(len(hist[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show()


pred = model(mnist_test_dataset.data.unsqueeze(1) / 255.)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Test accuracy: {is_correct.mean():.4f}')  # print test accuracy


fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    img = mnist_test_dataset[i][0][0, :, :]
    pred = model(img.unsqueeze(0).unsqueeze(1))
    y_pred = torch.argmax(pred)
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, y_pred.item(), 
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center', 
            transform=ax.transAxes)

plt.show()
''' 

