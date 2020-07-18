#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:40:29 2020

@author: vittorio
"""

import numpy as np
import gym


def FlatPolicySim(env, model, max_epoch, nTraj, size_input):
    
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    flag = np.empty((0,0),int)

    for episode in range(nTraj):
        done = False
        obs = env.reset()
        x = np.empty((0,2),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,0))
    
        for _ in range(0,max_epoch):
            # draw action
            prob_u = model(obs.reshape((1,size_input))).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled[0,:]))
            u_tot = np.append(u_tot,u)
        
            action = u*2
            obs, reward, done, info = env.step(action)
            x = np.append(x, obs.reshape((1,size_input)), axis=0)
        
            if done == True:
                u_tot = np.append(u_tot,0.5)
                break
        
        traj[episode][:] = x
        control[episode][:] = u_tot
        flag = np.append(flag,done)
        
    return traj, control, flag

def VideoFlatPolicy(environment, directory, model, size_input):
    env = gym.make(environment)
    # Record the environment
    env = gym.wrappers.Monitor(env, directory, force=True)

    for episode in range(1):
        done = False
        obs = env.reset()
        
        while not done: # Start with while True
                env.render()
                prob_u = model(obs.reshape((1,size_input))).numpy()
                prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
                for i in range(1,prob_u_rescaled.shape[1]):
                    prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
                draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
                u = np.amin(np.where(draw_u<prob_u_rescaled[0,:]))
        
                action = u*2
                obs, reward, done, info = env.step(action)
            
    env.close()
    
