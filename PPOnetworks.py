from Inputs import inputs
import torch
import torch.nn as nn
import torchvision
import torchvision as torchvision
import  numpy as np

class Actor(nn.Module):
    def __init__(self,inputsize,hiddensize,numlayers,numactions):
        super(Actor, self).__init__()
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.numlayers= numlayers
        self.numactions = numactions
        self.LSTM = nn.LSTM(self.inputsize,self.hiddensize,self.numlayers,batch_first = True,dropout = 0.2)
        #input shape req x = [batchsize,seq,inputsize]
        self.fc1= nn.Linear(self.hiddensize,112)
        self.fc2 = nn.Linear(112,56)
        self.fc3 = nn.Linear(56, 28)
        self.fc4 = nn.Linear(28,2*self.numactions)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim =-1)
        self.sig= nn.Sigmoid()
    def forward(self,x):
        x = torch.tensor(x,dtype = torch.float32)


        h0 = torch.zeros(self.numlayers,x.size(dim=0) ,self.hiddensize)
        c0 = torch.zeros(self.numlayers,x.size(dim=0) , self.hiddensize)

        out,_ = self.LSTM(x,(h0,c0))
        # out = [batchsize, seq,hiddensize]
        out = out[:,-1,:]
        # outputof only last timestep in sequence
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        self.relu(out)
        out = self.fc3(out)
        self.relu(out)
        out = self.fc4(out)
        mean, std = torch.tensor_split(out, 2, dim=-1)
        mean = self.soft(mean)
        std = self.sig(std)
        distribution = torch.distributions.normal.Normal(mean, std)
        action = self.soft(distribution.sample())
        prob = distribution.log_prob(action)


        return action , prob
    def save_weights(self):
        torch.save(self.state_dict(),self.savepointdirectory)

class Critic(nn.Module):
    def __init__(self,inputsize,hiddensize,numlayers,numactions):
        super(Critic, self).__init__()
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.numlayers= numlayers
        self.numactions = numactions
        self.LSTM = nn.LSTM(self.inputsize,self.hiddensize,self.numlayers,batch_first = True,dropout = 0.1)
        #input shape req x = [batchsize,seq,inputsize]
        self.fc1= nn.Linear(self.hiddensize+7,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8,self.numactions)
        self.relu = nn.ReLU()


    def forward(self,x,action):
        x = torch.tensor(x,dtype = torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        rank = torch.linalg.matrix_rank(x)

        action = torch.reshape(action,(x.size(0),x.size(-1))) #change 3 to variable


        h0 = torch.zeros(self.numlayers,x.size(dim=0) ,self.hiddensize)
        c0 = torch.zeros(self.numlayers,x.size(dim=0) , self.hiddensize)

        out,_ = self.LSTM(x,(h0,c0))
        # out = [batchsize, seq,hiddensize]
        out = out[:,-1,:]
        out = torch.cat((out, action), dim=1)
        # outputof only last timestep in sequence
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        return out
