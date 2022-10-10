import numpy as np
import torch

# class that handles input arrays for networks and functions that operate on input.

class inputs:
    def __init__(self, time, window, numcurrencies):
        self.time = time
        self.window = window  #size of the time window of which features of input timeseries are extracted
        self.numcurrencies = 7

    def inputarray(self, globalarray, init_time):


        inputs1 = globalarray[:, init_time - self.window:init_time]
        inputs = np.zeros([self.numcurrencies, self.window])
        for i in range(0, self.numcurrencies):
            inputs[i, :] = inputs1[i, :] / inputs1[i, self.window - 1]
        inputs = np.transpose(inputs)
        inputs = np.nan_to_num(inputs,nan = 0)
        return inputs

    def InputTensor(self, globalClose):
        self.inptensor = [self.inputarray(globalClose, self.time)]
        self.inptensor = np.array(self.inptensor)

        return self.inptensor

    def step(self, globalarray):
        state = np.array(self.inputarray(globalarray, self.time + 1))
        return state



    def reward(self, globalclose, wt):
        Yt = torch.tensor(globalclose[:, self.time-1] / globalclose[:, self.time ]) #calculates price change vector
        Yt[Yt != Yt] = 1   #finds nan in Yt and assigns value 0
        self.rt = np.dot(wt, Yt) # calculates return
        self.rt = np.log(self.rt)
        self.rt = torch.tensor(self.rt)
        return self.rt