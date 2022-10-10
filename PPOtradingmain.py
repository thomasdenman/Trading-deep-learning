#main file for PPO trading algorithm
#imports
from PPO import PPO_Agent
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from Inputs import inputs
import matplotlib.pyplot as plt
#importing data into dataframes
bitcoin = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Bitcoin.csv")
Dogecoin = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Dogecoin.csv")
Ethereum = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Ethereum.csv")
Litecoin = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Litecoin.csv")
USDcoin = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_USDcoin.csv")
Dogecoin = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_USDCoin.csv")
EOS = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_EOS.csv")
cardano = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Cardano.csv")
solana = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Solana.csv")
Dogecoin = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Dogecoin.csv")
Ethereum = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Ethereum.csv")
Litecoin = pd.read_csv("C:\\Users\\tomde\\Documents\\trading\\coin_Litecoin.csv")


#Input array of closing prices
globalClose = np.zeros((7,2991))
globalClose[0,:] = bitcoin.Close
globalClose[1,2991-len(Dogecoin.Close)-1:-1] = Dogecoin.Close
globalClose[2,:] = Litecoin.Close
globalClose[3,2991-len(solana.Close)-1:-1] = solana.Close
globalClose[4,2991-len(Ethereum.Close)-1:-1] = Ethereum.Close
globalClose[5,2991-len(cardano.Close)-1:-1] = cardano.Close
globalClose[6,2991-len(EOS.Close)-1:-1] = EOS.Close
globalClose = globalClose/globalClose[0,:]

#main loop
time = 0
maxtime = 600
initialtime =2200
n_steps = 0
N = 40
learnit = 0
agent = PPO_Agent(50)
for time in range(0,maxtime):
    done = False
    if time == maxtime:
        done = False
    env = inputs(initialtime+time, 50, 7)
    state = env.InputTensor(globalClose)

    action, value ,prob = agent.get_action(state)
    newstate = env.step(globalClose)
    reward = env.reward(globalClose,action)
    n_steps+=1
    agent.memory.store( action, reward, state, prob, value, done)
    if n_steps % N == 0:
        agent.learn()
        learnit +=1
        print(20*learnit)


#cacluate returns Prod t=1 to T ex(reward)
print(agent.loss_mem)
agent.loss_mem = torch.stack(agent.loss_mem).detach().numpy()
#plt.plot(agent.loss_mem)
agent.loss_mem1 = torch.stack(agent.loss_mem1).detach().numpy()
#plt.plot(agent.loss_mem1)
agent.memory.rewards_mem
returns = np.array(agent.epochrewards)
#returns = np.cumprod(np.exp(returns))
#plt.plot(returns)
rt = np.array(np.cumsum(agent.epochrewards))
benchmarkreturns = []
for time in range(0,maxtime):
    benchmark = inputs(initialtime+time,50,7)
    benchmarkreturns.append( benchmark.reward(globalClose,[3/7,1/7,1/7,1/7,1/7,0/7,0]))
    benchmark.step(globalClose)
benchmarkreturns = np.array(benchmarkreturns)
benchmarkreturns = np.cumsum(benchmarkreturns)

plt.plot(rt)
plt.plot(benchmarkreturns,color='green',linestyle = 'dashed')
plt.plot(rt-benchmarkreturns,color='red',linestyle = 'dashed')
plt.show()