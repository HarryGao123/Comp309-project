#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessory libraries
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
get_ipython().run_line_magic('matplotlib', 'inline')
from torchvision import models
import torch.optim as optim
import splitfolders  


# In[2]:


def set_device():
  if torch.cuda.is_available():
    dev = 'cuda:0'
  else:
    dev = 'cpu'
  return torch.device(dev)


# In[5]:


def evaluate_model_on_test(model,test):
  model.eval()
  correct_on_epoch = 0
  total = 0
  device = set_device()

  with torch.no_grad():
    for data in test:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      total += labels.size(0)

      outputs = model(images)

      _, predicted = torch.max(outputs.data, 1)

      correct_on_epoch += (predicted == labels).sum().item()

  epoch_acc = 100* correct_on_epoch / total

  print(' -Testing dataset. Got %d out of %d images correctly (%.3f%%)'
  % (correct_on_epoch, total, epoch_acc))


# In[4]:


def train_nn(model, train, test, criterion, optimizer, n_epochs):
  device = set_device()

  for epoch in range(n_epochs):
    print('Epoch number ', (epoch +1))
    model.train()
    running_loss = 0
    running_correct = 0
    total = 0

    for data in train:
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)
      total += labels.size(0)

      optimizer.zero_grad()

      outputs = model(images)

      _, predicted = torch.max(outputs.data, 1)

      loss = criterion(outputs, labels)

      loss.backward()

      optimizer.step()

      running_loss += loss.item()
      running_correct += (labels==predicted).sum().item()

    epoch_loss = running_loss / len(train)
    epoch_acc = 100 * running_correct / total
    print(' -train dataset. Got %d out of %d images correctly (%.3f%%)'
           % (running_correct, total, epoch_acc))
    
    evaluate_model_on_test(model, test)

  print('Done')
  return model      


# In[ ]:


if __name__ == '__main__': 
    transform = transforms.Compose([
    transforms.Resize([224,224]), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229,0.224, 0.225))])

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio("data", output="train_test", seed=42, ratio=(.8, .2), group_prefix=None) # default values
    
    train_dataset = ImageFolder('data/train_test/train',transform=transform)
    val_dataset = ImageFolder('data/train_test/val', transform=transform)
    
    # Set fixed random number seed
    torch.manual_seed(42)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    resnet = models.resnet101(pretrained=True)
    num_ftrs = resnet.fc.in_features
    num_class = 3
    resnet.fc = nn.Linear(num_ftrs, num_class)
    device = set_device()

    resnet = resnet.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr = 0.001, momentum=0.9)
    
    model_2 = train_nn(resnet, train_dl, val_dl, loss_fn, optimizer, 6)
    torch.save(model_2,'CNN_model.pth')

