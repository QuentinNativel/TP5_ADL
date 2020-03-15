#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pickle as pkl
import torch.nn as nn


# In[2]:


with open('traj.pkl', 'rb') as fp:
    traj = pkl.load(fp)


# In[3]:


class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.l1 = nn.Linear(3,10)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(10,10)
        self.l3 = nn.Linear(10,3)

    def forward(self, inputs):
        o1 = self.relu(self.l1(inputs))
        o2 = self.relu(self.l2(o1))
        return self.l3(o2)


# In[4]:


from torch.utils.data import DataLoader, random_split, TensorDataset

model = Model()

# Choose the hyperparameters for training:
num_epochs = 2000
batch_size = 10

# Use mean squared loss function
criterion = nn.MSELoss()
# Use SGD optimizer with a learning rate of 0.01
# It is initialized on our model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[5]:


def custom_train(num_epochs,criterion,optimizer,model,traj):
    model.train()
    train_error = []
    for epoch in range(num_epochs):

        pred = model(traj)
        mse = criterion(pred, traj)
        cust = custom_loss(pred, traj)

        l1_params = torch.cat([x.view(-1) for x in model.l1.parameters()])
        l2_params = torch.cat([x.view(-1) for x in model.l2.parameters()])
        l3_params = torch.cat([x.view(-1) for x in model.l3.parameters()])

        pen = torch.norm(l1_params,2) + torch.norm(l2_params,2) + torch.norm(l3_params, 2)

        loss = cust + 0.0 * mse + 0.01 * pen
        optimizer.zero_grad()
        #torch.autograd.grad(outputs=loss, inputs = traj)
        loss.backward()
        optimizer.step()

        epoch_average_loss = loss.item()
        train_error.append(epoch_average_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, epoch_average_loss))
    return  train_error


def custom_loss(pred, true):
    ret = ((1/12 * pred[:-5] - 2/3 * pred[1:-4] + 2/3 * pred[3:-2] -1/12 * pred[4:-1])
        - (1/12 * true[1:-4] - 2/3 * true[2:-3]+ 2/3 * true[4:-1] - 1/12 * true[5:]))**2
    return torch.sum(ret)


# In[6]:


train_error = custom_train(num_epochs, criterion, optimizer, model, traj)


# In[8]:


torch.save(model.state_dict(), 'model.pt')
