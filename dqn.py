#import gym
import collections
import random
import numpy as np
from DC_Working import model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import time
import sys

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

start = time.time()

count = 0
target_state = np.zeros(12)

df = pd.DataFrame(columns=['time','rpm'])
j = 0

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(12, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = ((x+1)/2)*255
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return np.random.randint(0,255)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)

        a = a.view(-1,1)

        a = a.float()
        s = s.float()
        """
        print(a)
        print(q_out)
        print(a.shape)
        print(q_out.shape)
        
        if q_out==a:
            q_a = q_out.gather(1,a)
        else:
            q_a = q_out

        """
        q_a = q_out
        #q_a = torch.gather(a,1,q_out)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        target = target.float()
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.detach(), reduce = False)
        loss = F.smooth_l1_loss(q_a, target.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def reward_calc(a,b): # Motor reward - converting from binary to decimal
        state1 = 0
        state2 = 0
        reward = 0
        for i in range(12):
            state1 += a[i]*(2**(11-i))
        for j in range(12):
            state2 += b[j]*(2**(11-j))
        if abs(state1 - state2) > 50:
            reward = -10*((state1-state2)**2)
        else:
            reward = 0
        return reward

def main():
    #env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 1
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    #target_state = np.array([0,1,1,1,1,1,0,1,0,0,0,0])
    done = False
    r = 0

    global count
    global j
    global start 

    for n_epi in range(500):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = np.array([0,1,1,1,1,1,1,1,1,1,1,1]) #env.reset()

        for t in range(10):
            done = False
            r = 0
            a = q.sample_action(torch.from_numpy(s.copy()).float(), epsilon)      
            #s_prime, r, done, info = env.step(a)
            if a > 255 :
                a = np.array([255])
            a = np.array([a])
            dec = model.predict(a)

            df.loc[j] = [time.time()-start] + [dec]
            j += 1
            """
            print('Current State = ',s)
            print('Action = ',a)
            print('Next State = ',dec)
            """
            dec = int(dec)
            i = 0
            s_prime = np.zeros(12)
            while(dec > 0):
                s_prime[i] = dec%2
                dec = int(dec/2)
                i += 1
            s_prime = np.flip(s_prime)


            r = reward_calc(s_prime,target_state)
            if r==0:
                done = True

            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            if np.array_equal(s,np.zeros(12)):
                s = np.array([0,1,1,1,1,1,1,1,1,1,1,1])


            score += r
            if done:
                print(a)               
                count += 1
                print('Yayy')
                print(count)
                """
                print(df)
                df.to_csv("DQN.csv")
                """
                sys.exit(0)
        
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)
        
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0

    print(count)
    #env.close()

if __name__ == '__main__':
    i = 0
    target = 0
    target = input("Enter rpm to be achieved (in dec) ")
    target = int(target)
    while(target > 0):
                target_state[i] = target%2
                target = int(target/2)
                i += 1
    target_state = np.flip(target_state)
    main()