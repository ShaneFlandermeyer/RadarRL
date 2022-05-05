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
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.models import Model
from radar_env import *

iterations=2500
def policy(observation, agent):
    '''
    Test function for environment testing.  DQN does not use this function
    '''
    global ind
    if agent == "Radar_0":        
        action = observation+1
    elif agent == "Comms_0":
        action = interference(observation, pattern, std=None, experiment='A')
    return action

#Function for translating int to a binary list of size N. 
def int_to_binlist(int_num, bin_size):
    binary = '{0:0%sb}'%(bin_size)
    return [int(i) for i in list(binary.format(int_num))]

#Function to translate a binary list to its integer equivalent
def binlist_to_int(bin_list):
    return int("".join(str(x) for x in bin_list), 2)
    
def interference(prev_state, pattern, std=None, experiment='sweep'):
    '''
    Function to generate the interference pattern for 2 experiments.
    Inputs: 
        prev_state = previous state from environment(must be in pattern)
        pattern = the states that the interferece will occupy
        std = for normal dist experiemnt this represents the standard deviation
              of normal dist for that pattern.
        experiemnts = label for the experiements 
            sweep = sweep single interference states
            normal_dist = the new interfernce position is based on a normal 
                 curve centered at the previous state.
    Output: The next pattern. 
    '''
    if experiment == 'sweep':
         #determine index of last pattern
         current_ind=pattern.index(prev_state)
         # if at the end of pattern loop to the front
         if current_ind == len(pattern)-1:
             return pattern[0]
         #sweep pattern
         else:
             return pattern[current_ind+1]    
    if experiment == 'normal_dist':
         #determine index of last pattern
         current_ind=pattern.index(prev_state)
         #normal dist to find new state index
         new_ind = round(np.random.normal(current_ind,std[current_ind]))
         # if dist is out of range then change to limit
         new_ind = 0 if new_ind < 0 else N-1 if new_ind > N-1 else new_ind
         return pattern[new_ind]
        
def max_reward(state, N):
    interf_ind = np.argmax(state)
    span = max(interf_ind,N-interf_ind-1)
    return (span-1)*10

def build_model(input_size,output_size,dense_layers, activation='elu', lrate=0.001):
    '''
    This function builds a single dense ANN to act as the DQN. 
    Inputs:
        input_size = size of input expectation
        output_size = size of expected output
        dense_layer = list of neuron in each layer.  
        activation = activation function for dense layers
        lrate = learning rate
    Output: model
    ''' 
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
    '''
    This function builds the specific architecture for position and span. 
    Inputs:
        input_size = size of input expectation
        output_size = size of expected output
        position_layer = list of neuron in each layer in position model. 
        span_layer = list of neuron in each layer in span model. 
        activation = activation function for dense layers
        lrate = learning rate
    Output: position model, span model
    '''
    position_model = build_model(input_size,output_size,position_layers, activation=activation, lrate=lrate)
    print(position_model.summary())
    span_model = build_model(input_size+output_size,output_size, span_layers, activation=activation, lrate=lrate) 
    print(span_model.summary())
    
    return position_model,span_model

RadarEnvironment = RadarEnvironment()
runs = 10
final_pattern_reward=[]
for r in range(runs):
    # experiment set-up and operation for DQN

    RadarEnvironment.reset()
    
    experiment = 'normal_dist'
    pattern = [1,2,4,8,16]  #interference pattern
    # pattern = [1,2,4,8,16,32,64,128,256,512]  #interference pattern
    std = [.5,.3,.2,.3,.5]   #standard deviation for experiment normal_dist
    # std = [.5,.3,.2,.3,.5,.5,.3,.2,.3,.5]
    gamma = .9              #discount factor for DQN learning
    eps = 1                 #starting epsilon for greedy epsilon selection
    eps_decay = 0.998       #decay factor for epsilon
    eps_min = .001          #minimum epsilon
    N= 5                    # number of sub-bands
    lrate = 0.05
    #build the position/span DQN
    position_model,span_model = build_agent_pair(input_size =N, output_size=N, position_layers=[128,64], 
                                                span_layers=[128,64], activation='elu', lrate=lrate)
    reward_totals =[]        #empty lists for appending
    reward_cumulative =[]
    state_totals = []
    action_totals = []
    optimal_reward_totals =[]
    
    #experiment iteration
    for i,agent in enumerate(RadarEnvironment.agent_iter()):
        #pull previous state
        observation, reward, done, info = RadarEnvironment.last()
        #each iterations radar agent update
        if agent == "Radar_0":   
            #previous state
            sys_obs = observation
            #state converted to binary list for DQN forward pass
            obs_list = int_to_binlist(observation, N)
            # position forward pass Q value list
            Q_pos = position_model.predict(np.reshape(obs_list,(1,N))).tolist()[0]
            #Input for span network as position output and state
            in_span = [*obs_list, *Q_pos] 
            # span forward pass Q value list
            Q_span = span_model.predict(np.reshape(in_span,(1,2*N))).tolist()[0]
            #epsilon greeding selection
            if random.random()<=eps and i<2*iterations-1001: #select random action state
                Ap = random.randrange(1,N)
                As = random.randrange(1,N)
            else: #select maximum Q value option. 
                Ap = np.argmax(position_model.predict(np.reshape(obs_list,(1,N))))+1
                As = np.argmax(span_model.predict(np.reshape(in_span,(1,2*N))))+1
            action_lst = [0]*N #empty final action list
            for j in range(N): #loop positions 
                if j>Ap-2 and j<Ap+As-1: #if position is true and within span set as 1
                    action_lst[j]=1
            #convert action to interger for the environment
            action = binlist_to_int(action_lst) if not done else None 
            #send Radar action to environment
            if not done:
                action_totals=np.append(action_totals, action_lst)
        #each iteration interference stat update
        elif agent == "Comms_0":
            # action determined by interference function
            action = interference(sys_obs, pattern, std=std, experiment=experiment) if not done else None 
            #send Interference action to environment
        RadarEnvironment.step(action)
        if agent == "Comms_0":
            #Update models using bellman equation
            #Pull next state after interfernce update
            observation, reward, done, info = RadarEnvironment.last()
            obs_list_next = int_to_binlist(observation, N)
            state_totals = np.append(state_totals, obs_list_next)
            if i< 2*iterations-1001:
                #Forward pass of S_t+1 
                Q_pos_next = position_model.predict(np.reshape(obs_list_next,(1,N)))
                in_span_next = [*obs_list, *Q_pos_next.tolist()[0]] 
                Q_span_next = span_model.predict(np.reshape(in_span_next,(1,2*N)))
                # set target as previous prediction
                target_pos = Q_pos
                # update the selection location with the bellman target
                target_pos[Ap-1] = reward+ gamma*np.max(Q_pos_next)
                # set target as previous prediction
                target_span = Q_span
                # update the selection location with the bellman target
                target_span[As-1] = reward+ gamma*np.max(Q_span_next)
                history_pos = position_model.fit(np.reshape(obs_list,(1,N)), np.reshape(target_pos,(1,N)),epochs=1, verbose=0)
                history_span = span_model.fit(np.reshape(in_span,(1,2*N)), np.reshape(target_span,(1,N)),epochs=1, verbose=0)
            reward_totals=np.append(reward_totals, reward)
            optimal_reward=max_reward(obs_list_next,N)
            optimal_reward_totals = np.append(optimal_reward_totals,optimal_reward)
            reward_cumulative = np.append(reward_cumulative, np.sum(reward_totals))
            if eps<=eps_min:
                eps=eps_min
            else:
                eps = eps*eps_decay
                
            
        
        # print('act:%s'%(action))
        if done:
            action_totals = np.resize(action_totals,(iterations,N))           
            state_totals = np.resize(state_totals,(iterations,N))
            plt.figure(2*r+1)
            plt.plot(reward_cumulative)
            plt.title("Experiment (%s_%s_%s) DQN"%(experiment,N,r+1))
            plt.ylabel('Cumulative Reward')
            plt.xlabel('Epochs')
            plt.grid(True)
            
            rewards_sum = np.sum(reward_totals.reshape(-1, N), axis=1)
            optimal_reward_sum = np.sum(optimal_reward_totals.reshape(-1, N), axis=1)
            mean_reward = np.mean(reward_totals[-500:])
            std_reward = np.std(reward_totals[-500:])
            mean_optimal = np.mean(optimal_reward_totals[-500:])
            std_optimal = np.std(optimal_reward_totals[-500:])
            
            plt.figure(2*r+2) 
            plt.plot(rewards_sum, label='DQN',color='r')
            plt.plot(optimal_reward_sum, label='optimal',color='b')
            plt.title("Experiment (%s_%s_%s) DQN"%(experiment,N,r+1))
            plt.ylabel('Reward total per pattern cycle')
            plt.xlabel('Pattern cycles')
            #plt.yticks(np.arange(-225,125,25))
            plt.legend()
            plt.grid(True)
            plt.savefig("Experiment_%s_%s_%s_DQN"%(experiment,N,r+1),bbox_inches='tight')
            
            final_pattern_reward=np.append(final_pattern_reward, rewards_sum[-1])
            results = {}
            results['reward_totals'] = reward_totals
            results['rewards_sum'] = rewards_sum
            results['action_totals'] = action_totals
            results['state_totals'] = state_totals
            results['optimal_rewards'] = optimal_reward_totals
            results['final_rewards_sum'] = rewards_sum[-1]
            with open("%s_%s_%s_results.pkl"%(experiment,N,r+1), "wb") as fp:
                pickle.dump(results, fp)
            break
mean_pattern_reward = np.mean(final_pattern_reward)
std_pattern_reward = np.std(final_pattern_reward)





            