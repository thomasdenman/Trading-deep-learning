# PPO algorithm trader
import numpy as np
import torch
import torch.optim as optim
from PPO_memory import replaytragectory
from PPOnetworks import Actor
from PPOnetworks import Critic


class PPO_Agent:
    def __init__(self,n_epochs):
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5
        self.n_epochs = n_epochs
        self.lr = 0.00001
        self.gamma = 0.99
        self.clip = 0.2
        self.gaelambda = 0.99
        self.batchsize = 20
        self.memory = replaytragectory(self.batchsize)
        self.loss_mem = []
        self.loss_mem1 = []

        self.epochrewards = []


        # Networks
        inputsize, hiddensize, numlayers, numactions = 7, 28 , 1, 7
        self.actor = Actor(inputsize, hiddensize, numlayers, numactions)
        self.critic = Critic(inputsize, hiddensize, numlayers, 1)


        self.actor_optim = optim.Adam(self.actor.parameters(), lr= self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr= self.lr)

    def save(self):
        self.memory.store(self, action, reward, state, prob, value, done)

    def get_action(self,Env):
        state = torch.tensor(Env)
        action , prob = self.actor.forward(state)
        value = self.critic(state,action)

        return action.detach().numpy(), value.detach().numpy() ,prob.detach().numpy()

    def learn(self):

        epochit = 0
        for  i in range(self.n_epochs):

            #action, reward, state, prob, value, done out order
            state_batch ,oldprob_batch ,values_batch, actions_batch ,rewards_batch, done_batch,batches = self.memory.generatebatches()
            values = values_batch
            advantage = np.zeros(len(rewards_batch))
            for t in range(len(rewards_batch)-1):
                discount = 1
                A_t = 0
                for k in range(t,len(rewards_batch)-1):
                    A_t = discount*(rewards_batch[k] + self.gamma*values[k+1])*(1-int(done_batch[k])) - values[k]
                    discount *= self.gamma*self.gaelambda
                advantage[t] = A_t
            advantage = torch.tensor(advantage,dtype = torch.float64)
           # values = torch.tensor(values)
            for batch in batches:
                batch = batch.tolist()

                states = torch.tensor(state_batch[batch] ,dtype = torch.float64)
                states = torch.squeeze(states)
                old_probs = torch.squeeze(torch.tensor(oldprob_batch[batch]))
                actions = torch.tensor(actions_batch[batch])

                action , prob = self.actor.forward(states)
                criticvalue = self.critic.forward(states,actions)


                new_prob = prob
                probabilityratio = torch.exp(torch.sum(new_prob,1)) / torch.exp(torch.sum(old_probs,1))
                weightedprob = advantage[batch] * probabilityratio
                weightedclippedprob = torch.clamp(weightedprob,1-self.clip,1+self.clip)*advantage[batch]

                self.policyloss = torch.min(weightedprob , weightedclippedprob).mean()
                returns = advantage[batch]+values[batch]
                self.criticloss = (returns-criticvalue)**2
                self.criticloss = self.criticloss.mean()

                self.loss = self.criticloss + self.policyloss
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                self.loss.backward()
               # self.criticloss.backward()
               # policyloss.backward()
                self.actor_optim.step()
                self.critic_optim.step()
            epochit +=1
            print(epochit,self.policyloss,self.criticloss)
           # print(actions_batch)

            self.loss_mem.append(self.criticloss)
            self.loss_mem1.append(self.policyloss)
        self.epochrewards.append(rewards_batch)

        self.memory.clear_mem()















