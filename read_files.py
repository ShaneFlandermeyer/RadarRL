# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:16:41 2022

@author: Geoffrey Dolinger
"""
import pickle
import fnmatch
import os
import numpy as np

def read_all_rotations(dirname,filebase):
    '''Read results from dirname from files matching filebase'''

    # The set of files in the directory
    files = fnmatch.filter(os.listdir(dirname),filebase)
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)
    return results

dirname = "final_runs"
filebase = 'sweep_10_*_results.pkl'
results = read_all_rotations(dirname,filebase)

mean_testing_total = []
std_testing_total = []
mean_opt_total = []
std_opt_total = []

for r in results:
    rewards = r['reward_totals']
    mean_testing = np.mean(rewards[-500:])
    std_testing = np.std(rewards[-500:])
    mean_testing_total = np.append(mean_testing_total,mean_testing)
    std_testing_total = np.append(std_testing_total,std_testing)
    
    optimal = r['optimal_rewards']
    mean_opt = np.mean(optimal[-500:])
    std_opt = np.std(optimal[-500:])
    mean_opt_total = np.append(mean_opt_total,mean_opt)
    std_opt_total = np.append(std_opt_total,std_opt)

final_mean = np.mean(mean_testing_total)
final_std = np.sqrt(np.mean(std_testing_total**2))
final_opt_mean= np.mean(mean_opt_total)
final_opt_std= np.sqrt(np.mean(std_opt_total**2))