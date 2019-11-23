import os
import numpy as np
import math
import sys
import serial
from time import sleep
#from keras.models import model_from_json
#from DC_Working import model

count = 0
target_state = np.zeros(12)

class ARSTrainer_own():
    # Class where the agent learns to play the game.
    def __init__(self,
                 nb_steps=5,
                 episode_length=5,
                 learning_rate=0.02,
                 num_deltas=2,
                 num_best_deltas=2,
                 noise=0.1,
                 record_every=50,
                 seed=1946):
        
        self.nb_steps = nb_steps
        np.random.seed(seed)
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        self.noise = noise       
        self.output_size = 1 # Pwm as action
        self.input_size = 12  # Rpm as states - converted to binary 
        self.record_every = record_every
        self.theta = np.random.randn(self.output_size,self.input_size)
        self.state = np.array([0,1,1,1,1,1,1,1,1,1,1,1])   
        self.seri = serial.Serial('/dev/ttyUSB0',115200)
        self.seri.close()
        self.seri.open() 
        self.seri.write(serial.to_bytes([251,000]))


    def update(self,
                learning_rate,
                num_best_deltas,
                sigma_rewards,
                rollouts):
        """
        Helper function to update the value of the parameter theta

        Args:
        learning rate : rate at which updation is done
        num_best_deltas : the first best deltas which we consider for rollouts
        sigma_rewards : Std.deviation of the total rewards ( r_pos + r_neg)
        rollouts : Tuple containing (r_pos,r_neg,deltas)
        """
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        self.theta += learning_rate/(num_best_deltas*sigma_rewards)*step
    
    def gen_random_deltas(self):
        """
        Helper function to generate random sample of delta
        
        """
        print('GEt random delta')
        delta = np.random.uniform(low=0.0,high=200.0,size=(self.num_deltas,self.output_size,self.input_size))
        return delta

    def select_action(self,input,direction=None,delta=None):
        """
        Helper function to select the action according to our policy
        
        Args:
        input : The state is the input
        direction : direction in which the update should be done in theta
        delta : Randomly generated matrix to add noise to theta
        """
        if direction is None or delta is None:
            return self.theta.dot(input)
        elif direction == '+':
            print('Entered here')
            return ((self.theta + self.noise*delta).dot(input) )
        elif direction == '-':
            print('Entered here')
            return ((self.theta - self.noise*delta).dot(input) )

    def reward_calc(self,a,b): # Motor reward - converting from binary to decimal
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


    def execute(self,direction=None,delta=None):
        """
        Helper function to execute the selected action 
        
        Args:
        direction :- direction in which the update should be done in theta.
        delta :- Randomly generated matrix to add noise to theta.
        """
        global count
        global target_state
        #target_state = np.array([0,1,0,1,1,1,0,1,1,1,0,0]) # 1500 in rpm        
        sum_r = 0
        num_play = 0
        done = False
        while not done and num_play <= self.episode_length :            
            action = self.select_action(self.state,direction,delta) 
            print('Current State = ',self.state)
            print('action = ',action)
            if action > 255 :
                action = np.array([255])
            if action < 50 :
                action = np.array([50])
            
            """
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from
            _json(loaded_model_json)
            """
            action = int(action)
            print(action)
            self.seri.write(serial.to_bytes([254,action]))
            sleep(0.5)
            self.seri.write(serial.to_bytes([255]))
            data = self.seri.readline(2).decode()
            dec = ord(data[1])*35.9281# model.predict(action)
            print('Next State = ',dec)
            dec = int(dec)
            ##Enter here as model.predict
            i = 0
            next_state = np.zeros(12)
            while(dec > 0):
                next_state[i] = dec%2
                dec = int(dec/2)
                i += 1
            next_state = np.flip(next_state)


            reward = self.reward_calc(next_state,target_state)
            if reward==0:
                done = True
            sum_r+=reward
            self.state = next_state
            if np.array_equal(self.state,np.zeros(12)):
                self.state = np.array([0,1,1,1,1,1,1,1,1,1,1,1])
            num_play+=1
            if done: 
                print(action)
                action = int(action)               
                count += 1
                print('Yayy')
                print(count)                
                self.seri.write(serial.to_bytes([254,action]))
                try:
                    while(1):
                        self.seri.write(serial.to_bytes([255]))
                        data = self.seri.readline(2).decode()
                        print(ord(data[1])*35.9281)
                except KeyboardInterrupt:
                    self.seri.write(serial.to_bytes([254,000]))
                    self.seri.close()
                    sys.exit(0)
        return sum_r
    
    def roll_best(self,rollouts):
        best_roll = np.zeros((self.num_best_deltas,2),dtype=np.int)
        #best_roll = []
        for k,[r1,r2,delta] in enumerate(rollouts):
            best_roll[k][1] = max(r1,r2)
            best_roll[k][0] = int(k)
        best_roll = best_roll[best_roll[:,1].argsort()[::-1]]
        #print(best_roll)
        return [rollouts[i] for i in best_roll[:self.num_best_deltas,0]]


    def train(self):
        """
        Function to train the agent to play the game
        """
        
        for i in range(self.nb_steps):
            deltas = self.gen_random_deltas()
            r_pos = np.zeros(self.num_deltas) # Initialising r[+]
            r_neg = np.zeros(self.num_deltas) # Initialising r[-]

            # Calculating positive and negative rewards
            for k in range(self.num_deltas):
                r_pos[k] = self.execute(direction="+", delta=deltas[k])
                r_neg[k] = self.execute(direction="-", delta=deltas[k])

            # Calculate Standard Deviation of the rewards
            self.sigma_rewards = np.array(r_pos + r_neg ).std()

            rollouts = []

            for k,(r1,r2,delta) in enumerate(zip(r_pos,r_neg,deltas)):
                rollouts.append([r1,r2,delta])

            rollouts = self.roll_best(rollouts)

            # Update the parameter theta
            self.update(self.learning_rate,self.num_best_deltas,self.sigma_rewards,rollouts)

            reward_evaluation = self.execute()
            print('Episode Number:', i, 'Reward:', reward_evaluation)


if __name__ == '__main__':    
    #global target
    #global target_state
    i = 0
    target = 0
    target = input("Enter rpm to be achieved (in dec) ")
    target = int(target)
    while(target > 0):
                target_state[i] = target%2
                target = int(target/2)
                i += 1
    target_state = np.flip(target_state)

    agent = ARSTrainer_own() # Calls init method
    agent.train()