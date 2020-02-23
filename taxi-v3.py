import numpy as np
import gym
import matplotlib.pyplot as plt
import random


env=gym.make('Taxi-v3').env

#Q table
q_table=np.zeros([env.observation_space.n, env.action_space.n])

#HyperParameters

alpha= 0.1
gamma = 0.95
epsilon = 0.2

#Plotting matrix

reward_list = []
dropout_list = []

episode = 100000

for i in range(1, episode):
    #Initialize the environment
    state=env.reset()
    reward_count = 0
    dropout_count = 0
    while True:
        #exploit and explore
        if random.uniform(0,1)<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        # action process and reward 
        next_state, reward, done, _ = env.step(action)
        
        old_value=q_table[state,action]
        next_max = np.max(q_table[next_state])
        #Q LEARNING FUNCTION
        next_value = (1-alpha)*old_value + alpha*(reward+gamma*next_max)
        
        #Q table update
        q_table[state, action] = next_value
        
        #state update
        state= next_state
        
        #find wrong dropouts
        if reward == -10:
            dropout_count+=1
        reward_count+=reward
        if done:
            break
    if i%10==0:  
        dropout_list.append(dropout_count)
        reward_list.append(reward_count)
        print("Episode count: {}, reward: {}, wrong dropout: {}".format(i, reward_count, dropout_count))
        
    