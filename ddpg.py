import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DC_Working import model

import pandas as pd
import time
import sys

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

count = 0
target_state = np.zeros(12)

start = time.time()

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

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = ((torch.tanh(self.fc_mu(x)) + 1)/2)*255 
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        
        self.fc_s = nn.Linear(12, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32,1)

    def forward(self, x, a):
        x = x.float()
        a = a.float()
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        #print('h1',h1.shape)
        #print('h2',h2.shape)
        h2 = h2.reshape(h1.shape)
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime))
    a = a.float()
    s = s.float()
    target = target.float()
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


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
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    #target_state = np.array([0,1,1,1,1,1,0,1,0,0,0,0]) # 1500 in rpm 11111010000
    done = False

    score = 0.0
    print_interval = 20

    global count 
    global j
    global start

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(500):
        s = np.array([0,1,1,1,1,1,1,1,1,1,1,1]) #env.reset()
        #s = s.astype(int)

        for t in range(10): # maximum length of episode is 200 for Pendulum-v0
            done = False
            r = 0
            a = (mu(torch.from_numpy(s.copy()).float())) 
            a = a.item() + ou_noise()[0]
            a = int(a)
            a = np.array([a])
            #s_prime, r, done, info = env.step([a])
            if a > 255 :
                a = np.array([255])
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



            memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime

            if np.array_equal(s,np.zeros(12)):
                s = np.array([0,1,1,1,1,1,1,1,1,1,1,1])

            if done:
                print(a)               
                count += 1
                print('Yayy')
                print(count)
                """
                print(df)
                df.to_csv("DDPG.csv")
                """
                sys.exit(0)              
                
        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    print(count)

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