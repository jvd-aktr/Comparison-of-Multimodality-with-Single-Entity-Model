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
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




## Step 1: loading and preprocessing MNIST dataset

path_train = r"C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\ImgTxtMapped_data_total\mnist_likedata_total\train_updated.csv"
path_test = r"C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\ImgTxtMapped_data_total\mnist_likedata_total\test_updated.csv"
transform = transforms.Compose([transforms.ToTensor()])

# For train dataset
df_train = pd.read_csv(path_train) 

imagePixelRow = 64
imagePixelCol = 64

train_dataset = []
 
for index, row in df_train.iterrows():
    i = 0
    singlePixelRow = []
    oneImageAllRows = []
    label = -1
    for col in row:
        if i!=0:
            singlePixelRow.append(col/255)
            if len(singlePixelRow) == imagePixelRow:
                oneImageAllRows.append(singlePixelRow)
                singlePixelRow = []
        else:
            label = col
            i+=1
    tensor_3d = torch.tensor([oneImageAllRows])
    train_dataset.append((tensor_3d, label))

mnist_dataset = train_dataset

mnist_valid_dataset = torch.utils.data.Subset(mnist_dataset,torch.arange(2000))
mnist_train_dataset = torch.utils.data.Subset(mnist_dataset,torch.arange(2000, len(mnist_dataset)))

# For test dataset
df_test = pd.read_csv(path_test) 

imagePixelRow = 64
imagePixelCol = 64

test_dataset = []

for index, row in df_test.iterrows():
    i = 0
    singlePixelRow = []
    oneImageAllRows = []
    label = -1
    for col in row:
        if i!=0:
            singlePixelRow.append(col/255)
            if len(singlePixelRow) == imagePixelRow:
                oneImageAllRows.append(singlePixelRow)
                singlePixelRow = []
        else:
            label = col
            i+=1
    tensor_3d = torch.tensor([oneImageAllRows])
    test_dataset.append((tensor_3d, label))
    
mnist_test_dataset = test_dataset

print('number of items in mnist_dataset:', len(mnist_dataset))
print('number of items in mnist_train_dataset:', len(mnist_train_dataset))
print('number of items in mnist_valid_dataset:', len(mnist_valid_dataset))
print('number of items in mnist_test_dataset:', len(mnist_test_dataset))


## Construct the data loader
batch_size = 64
torch.manual_seed(1)

train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

print(type(mnist_dataset))
print(len(mnist_dataset))
# print(mnist_dataset[0])


## Construct a CNN in PyTorch
model = nn.Sequential()

# tune the model by changing the parameters
model.add_module('conv1',nn.Conv2d(in_channels=1, 
                                   out_channels=32, 
                                   kernel_size=5, 
                                   padding=2))
model.add_module('relu1',nn.ReLU())
model.add_module('pool1',nn.MaxPool2d(kernel_size=2))

model.add_module('conv2',nn.Conv2d(in_channels=32, 
                                   out_channels=64, 
                                   kernel_size=5, 
                                   padding=2))
model.add_module('relu2',nn.ReLU())
model.add_module('pool2',nn.MaxPool2d(kernel_size=2))

# model.add_module('conv3',nn.Conv2d(in_channels=64,out_channels=8,kernel_size=4, stride=4, padding=0))
# model.add_module('relu3',nn.ReLU())
# model.add_module('pool3',nn.MaxPool2d(kernel_size=5, stride=5))

x = torch.ones((64, 1, 64, 64))
model(x).shape

model.add_module('flatten', nn.Flatten())
x = torch.ones((64, 1, 64, 64))
model(x).shape

model.add_module('fc1', nn.Linear(16384, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout1', nn.Dropout(p=0.9))

model.add_module('fc2', nn.Linear(1024, 256))
model.add_module('relu4', nn.ReLU())
model.add_module('dropout2', nn.Dropout(p=0.7))

model.add_module('fc3', nn.Linear(256, 64))
model.add_module('relu5', nn.ReLU())
model.add_module('dropout3', nn.Dropout(p=0.5))

model.add_module('fc4', nn.Linear(64, 2))

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
num_epochs = 25
device = 'cpu'
hist = train(model, num_epochs, train_dl, valid_dl)


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

'''
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
