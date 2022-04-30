import mdptoolbox
import mdptoolbox.example
import numpy as np
import scipy.constants as sc
import itertools
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib import animation

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.models import Model
from radar_env import *




RadarEnvironment = RadarEnvironment()
observation=RadarEnvironment.reset()
RadarEnvironment.render()
print(observation)

pattern = [1,2,4,8,16]
index = [1,2,3,4,5]
std = [.1,.2,.3,1,.5]
ind = 0
gamma = .8
eps = 1
eps_decay = 0.9975
eps_min = .001
N=5
def policy(observation, agent):
    global ind
    if agent == "Radar_0":        
        action = observation+1
    elif agent == "Comms_0":
        action = interference(observation, pattern, std=None, experiment='A')
    return action

def int_to_binlist(int_num, bin_size):
    binary = '{0:0%sb}'%(bin_size)
    return [int(i) for i in list(binary.format(int_num))]

def binlist_to_int(bin_list):
    return int("".join(str(x) for x in bin_list), 2)
    
def interference(prev_state, pattern, std=None, experiment='A'):
     if experiment == 'A':
         current_ind=pattern.index(prev_state)
         if current_ind == len(pattern)-1:
             return pattern[0]
         else:
             return pattern[current_ind+1]    
     if experiment == 'B':
         current_ind=pattern.index(prev_state)
         new_ind = round(np.random.normal(current_ind,std[current_ind]))
         new_ind = 0 if new_ind < 0 else N-1 if new_ind > N-1 else new_ind
         return pattern[new_ind]
        

def build_model(input_size,output_size,dense_layers, activation='elu', lrate=0.001):
     input_tensor = Input(shape =input_size,name = "input")
     tensor=input_tensor
     for i, dense in enumerate(dense_layers):
          #Add Dense layers based on input parameters for each layer
         tensor=Dense(dense, use_bias=True, name="FC%02d"%(i), 
                    activation=activation)(tensor)
    
     output_tensor = Dense(output_size, use_bias=True, name="Output", activation="linear")(tensor)
     
     opt = tf.keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
     model=Model(inputs=input_tensor, outputs=output_tensor)
     model.compile(loss='mse', optimizer=opt,metrics=["accuracy"]) 
     
     return model
 
def build_agent_pair(input_size, output_size, position_layers, 
                     span_layers, activation='elu', lrate=0.001):  
    position_model = build_model(input_size,output_size,position_layers, activation=activation, lrate=lrate)
    print(position_model.summary())
    span_model = build_model(input_size+output_size,output_size, span_layers, activation=activation, lrate=lrate) 
    print(span_model.summary())
    
    return position_model,span_model
  
RadarEnvironment.reset()
position_model,span_model = build_agent_pair(input_size =N, output_size=N, position_layers=[128,64], 
                                            span_layers=[128,64], activation='elu', lrate=0.01)
reward_total =[]
reward_cumulative =[]
for i,agent in enumerate(RadarEnvironment.agent_iter()):
    observation, reward, done, info = RadarEnvironment.last()
    # print('agent:%s'%(agent))
    # print('obs:%s'%(observation))
    # print('rew:%s'%(reward))
    if agent == "Radar_0":
        
        sys_obs = observation
        obs_list = int_to_binlist(observation, N)
        Q_pos = position_model.predict(np.reshape(obs_list,(1,N))).tolist()[0]
        in_span = [*obs_list, *Q_pos] 
        Q_span = span_model.predict(np.reshape(in_span,(1,2*N))).tolist()[0]
        if random.random()<=eps:
            Ap = random.randrange(1,N)
            As = random.randrange(1,N)
        else:
            Ap = np.argmax(position_model.predict(np.reshape(obs_list,(1,N))))+1
            As = np.argmax(span_model.predict(np.reshape(in_span,(1,2*N))))+1
        action_lst = [0]*N
        for j in range(N):
            if j>Ap-2 and j<Ap+As-1:
                action_lst[j]=1
        action = binlist_to_int(action_lst) if not done else None 
        RadarEnvironment.step(action)
    elif agent == "Comms_0":
        action = interference(sys_obs, pattern, std=std, experiment='A') if not done else None 
        RadarEnvironment.step(action)
        #Update models
        observation, reward, done, info = RadarEnvironment.last()
        obs_list_next = int_to_binlist(observation, N)
        Q_pos_next = position_model.predict(np.reshape(obs_list_next,(1,N)))
        in_span_next = [*obs_list, *Q_pos_next.tolist()[0]] 
        Q_span_next = span_model.predict(np.reshape(in_span_next,(1,2*N)))
        target_pos = Q_pos
        target_pos[Ap-1] = reward+ gamma*np.max(Q_pos_next)
        target_span = Q_span
        target_span[As-1] = reward+ gamma*np.max(Q_span_next)
        history_pos = position_model.fit(np.reshape(obs_list,(1,N)), np.reshape(target_pos,(1,N)),epochs=1, verbose=0)
        history_span = span_model.fit(np.reshape(in_span,(1,2*N)), np.reshape(target_span,(1,N)),epochs=1, verbose=0)
        reward_total=np.append(reward_total, reward)
        reward_cumulative = np.append(reward_cumulative, np.sum(reward_total))
        if eps<=eps_min:
            eps=eps_min
        else:
            eps = eps*eps_decay
            
        
    
    # print('act:%s'%(action))
    if i>3989 and agent== "Comms_0":
        RadarEnvironment.render()
        print(RadarEnvironment._cumulative_rewards)
        plt.plot(reward_cumulative)
        plt.title("Experiment A DQN")
        plt.ylabel('Cumulative Reward')
        plt.xlabel('Epochs')
        plt.grid(True)
        
       
        for p in pattern:
           state = int_to_binlist(p, N)
           print('State:  %s'%(state))
           Q_pos = position_model.predict(np.reshape(state,(1,N))).tolist()[0]
           in_span = [*state, *Q_pos] 
           Q_span = span_model.predict(np.reshape(in_span,(1,2*N))).tolist()[0]
           Ap = np.argmax(position_model.predict(np.reshape(state,(1,N))))+1
           As = np.argmax(span_model.predict(np.reshape(in_span,(1,2*N))))+1 
           action_lst = [0]*N
           for j in range(N):
               if j>Ap-2 and j<Ap+As-1:
                   action_lst[j]=1
           action = binlist_to_int(action_lst)
           
           print('Action: %s'%(action_lst))
           
            