#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


def set_device():
  if torch.cuda.is_available():
    dev = 'cuda:0'
  else:
    dev = 'cpu'
  return torch.device(dev)

device = set_device()


# In[ ]:


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
      


# In[ ]:


def test():
    
    #set up model
    model = torch.load('CNN_model.pth',map_location=device)
    
    # transoform images
    transform = transforms.Compose([
    transforms.Resize([224,224]), 
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), (0.229,0.224, 0.225))])
    
    # read images from 'testdata'
    test_data = ImageFolder('testdata',transform=transform)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    # get the accuracy
    evaluate_model_on_test(model, test_dl)
    


# In[ ]:


# run it
if __name__ == '__main__': 
    test()

