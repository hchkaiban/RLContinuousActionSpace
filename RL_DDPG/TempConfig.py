#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:07:28 2017
@author: hc
Global config 
"""
import os
import pickle
import matplotlib.pyplot as plt

ModelsPath = "KerasModels/"

Temporal_Buffer = 4       
        
        
def save_DDQL(Path, Name, agent):
    ''' Saves actions, rewards and states (images) in DataPath'''
    if not os.path.exists(Path):
        os.makedirs(Path)
    agent.model.save(Path+Name)
    print(Name, "saved")
    print('...')
    
    #dump_pickle(agent.memory, Path+Name+'Memory')
    
    #dump_pickle([agent.epsilon, agent.steps, agent.brain.opt.get_config()], Path+Name+'AgentParam')
    #dump_pickle(R, Path+Name+'Rewards')
    print('Memory pickle dumped')


def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def Plot(array, save):   
    ''' Plot 1D array '''    
    plt.plot(array)
    axes = plt.gca()
    axes.set_xlim([0,1500])
    axes.set_ylim([-200,200])
    plt.xlabel('Episodes')
    plt.ylabel('array')
    plt.title('plot')
    plt.grid(True)
    if save:
        plt.savefig("Rplot.png")
    plt.show()