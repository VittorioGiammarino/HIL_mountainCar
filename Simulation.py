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
    env = gym.wrappers.Monitor(env, directory, resume=True)

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
    
def HierarchicalPolicySim():
    traj = [[None]*1 for _ in range(nTraj)]
    control = [[None]*1 for _ in range(nTraj)]
    Option = [[None]*1 for _ in range(nTraj)]
    Termination = [[None]*1 for _ in range(nTraj)]
    flag = np.empty((0,0),int)
    
    for episode in range(nTraj):
        done = False
        obs = env.reset()
        x = np.empty((0,0),int)
        x = np.append(x, obs.reshape((1,size_input)), axis=0)
        u_tot = np.empty((0,0))
        o_tot = np.empty((0,0),int)
        b_tot = np.empty((0,0),int)
        
        # Initial Option
        prob_o = mu
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[0]):
            prob_o_rescaled[i]=prob_o_rescaled[i]+prob_o_rescaled[i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled))
        o_tot = np.append(o_tot,o)
        
        # Termination
        state = stateSpace[x[0],:].reshape(1,len(stateSpace[x[0],:]))
        prob_b = NN_termination(np.append(state,[[o]], axis=1)).numpy()
        prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
        for i in range(1,prob_b_rescaled.shape[1]):
            prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
        draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
        b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
        b_tot = np.append(b_tot,b)
        if b == 1:
            b_bool = True
        else:
            b_bool = False
        
        o_prob_tilde = np.empty((1,option_space))
        if b_bool == True:
            o_prob_tilde = NN_options(state).numpy()
        else:
            o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
            o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
        prob_o = o_prob_tilde
        prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
        for i in range(1,prob_o_rescaled.shape[1]):
            prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
        draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
        o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
        o_tot = np.append(o_tot,o)
        
        for k in range(0,max_epoch):
            state = stateSpace[x[k],:].reshape(1,len(stateSpace[x[k],:]))
            # draw action
            prob_u = NN_actions(np.append(state,[[o]], axis=1)).numpy()
            prob_u_rescaled = np.divide(prob_u,np.amin(prob_u)+0.01)
            for i in range(1,prob_u_rescaled.shape[1]):
                prob_u_rescaled[0,i]=prob_u_rescaled[0,i]+prob_u_rescaled[0,i-1]
            draw_u=np.divide(np.random.rand(),np.amin(prob_u)+0.01)
            u = np.amin(np.where(draw_u<prob_u_rescaled)[1])
            
            # given action, draw next state
            x_k_possible=np.where(P[x[k],:,int(u)]!=0)
            prob = P[x[k],x_k_possible[0][:],int(u)]
            prob_rescaled = np.divide(prob,np.amin(prob))
            
            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            x = np.append(x, x_k_possible[0][index_x_plus1])
            u_tot = np.append(u_tot,u)
            
            if x[k+1]==terminal_state:
                state = stateSpace[terminal_state,:].reshape(1,len(stateSpace[terminal_state,:]))
                u_tot = np.append(u_tot,np.argmax(NN_actions(np.append(state,[[o]], axis=1)).numpy()))
                break
            
            # Select Termination
            # Termination
            state_plus1 = stateSpace[x[k+1],:].reshape(1,len(stateSpace[x[k+1],:]))
            prob_b = NN_termination(np.append(state_plus1,[[o]], axis=1)).numpy()
            prob_b_rescaled = np.divide(prob_b,np.amin(prob_b)+0.01)
            for i in range(1,prob_b_rescaled.shape[1]):
                prob_b_rescaled[0,i]=prob_b_rescaled[0,i]+prob_b_rescaled[0,i-1]
            draw_b = np.divide(np.random.rand(), np.amin(prob_b)+0.01)
            b = np.amin(np.where(draw_b<prob_b_rescaled)[1])
            b_tot = np.append(b_tot,b)
            if b == 1:
                b_bool = True
            else:
                b_bool = False
        
            o_prob_tilde = np.empty((1,option_space))
            if b_bool == True:
                o_prob_tilde = NN_options(state_plus1).numpy()
            else:
                o_prob_tilde[0,:] = zeta/option_space*np.ones((1,option_space))
                o_prob_tilde[0,o] = 1 - zeta + zeta/option_space
            
            prob_o = o_prob_tilde
            prob_o_rescaled = np.divide(prob_o, np.amin(prob_o)+0.01)
            for i in range(1,prob_o_rescaled.shape[1]):
                prob_o_rescaled[0,i]=prob_o_rescaled[0,i]+prob_o_rescaled[0,i-1]
            draw_o=np.divide(np.random.rand(), np.amin(prob_o)+0.01)
            o = np.amin(np.where(draw_o<prob_o_rescaled)[1])
            o_tot = np.append(o_tot,o)
            
        
        traj[t][:] = x
        control[t][:]=u_tot
        Option[t][:]=o_tot
        Termination[t][:]=b_tot
        
        if x[-1]==terminal_state:
            success = 1
        else:
            success = 0
                
        flag = np.append(flag,success)
        
    return traj, control, Option, Termination, flag                
    
