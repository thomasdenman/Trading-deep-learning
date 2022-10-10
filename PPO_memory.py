import numpy as np
import torch
class replaytragectory:
    def __init__(self,batchsize):
        self.state_mem = []
        self.probs_mem = []
        self.values_mem = []
        self.actions_mem = []
        self.rewards_mem = []
        self.terminal_mem = []
        self.batchsize = batchsize
        self.mem_counter =0
    def generatebatches(self):
        n_states = len(self.state_mem)
        batchstart = np.arange(0,n_states,self.batchsize,dtype = int)
        index = np.arange(0,n_states)
        np.random.shuffle(index)
        batches = [index[i:i+self.batchsize] for i in batchstart]


        return np.array(self.state_mem),\
        np.array(self.probs_mem),\
        np.array(self.values_mem),\
        np.array(self.actions_mem),\
        np.array(self.rewards_mem),\
        np.array(self.terminal_mem),\
        batches



    def store(self, action, reward, state, prob, value,done):


        self.state_mem.append(state)
        self.actions_mem.append(action)
        self.probs_mem.append(prob)
        self.values_mem.append(value)
        self.rewards_mem.append(reward)
        self.terminal_mem.append(done)
        self.mem_counter+=1

    def clear_mem(self):
        self.state_mem = []
        self.probs_mem = []
        self.values_mem = []
        self.actions_mem = []
        self.rewards_mem = []
        self.terminal_mem = []