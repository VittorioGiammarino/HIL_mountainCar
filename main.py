#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:52:47 2020

@author: vittorio
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
import Simulation as sim
import BehavioralCloning as bc
import HierachicalImitationLearning as hil
import concurrent.futures
import gym

# %% map generation 
bc_data_dir = 'data'
TrainingSet, labels = hil.PreprocessData(bc_data_dir)

TrainingSet = TrainingSet[0:2000,:]
labels = labels[0:2000]
# %%
fig = plt.figure()
plot_action = plt.scatter(TrainingSet[:,0], TrainingSet[:,1], c=labels, marker='x', cmap='winter');
cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('Expert_state_action_distribution.eps', format='eps')
plt.show()


# %% Behavioral Cloning vs Baum-Welch algorithm
option_space = 2
action_space = 2
termination_space = 2
size_input = TrainingSet.shape[1]

NN_options = hil.NN_options(option_space, size_input)
NN_actions = hil.NN_actions(action_space, size_input)
NN_termination = hil.NN_termination(termination_space, size_input)

N=10 #Iterations
zeta = 0.1 #Failure factor
mu = np.ones(option_space)*np.divide(1,option_space) #initial option probability distribution

gain_lambdas = np.logspace(-2, 3, 7, dtype = 'float32')
gain_eta = np.logspace(-2, 3, 7, dtype = 'float32')

list_triple = []
validation = []
k=0

Triple = hil.Triple(NN_options, NN_actions, NN_termination)

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 1000

averageHIL = np.empty((0))
success_percentageHIL = np.empty((0))

for i in range(len(gain_lambdas)):
    for j in range(len(gain_eta)):
        T = TrainingSet.shape[0]
        TrainingSetTermination = hil.TrainingSetTermination(TrainingSet, option_space, size_input)
        TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels, size_input)
        lambdas = tf.Variable(initial_value=gain_lambdas[i]*tf.ones((option_space,)), trainable=False)
        eta = tf.Variable(initial_value=gain_eta[j], trainable=False)
        NN_Termination, NN_Actions, NN_Options = hil.BaumWelch(labels, TrainingSet, size_input, action_space, option_space, termination_space, 
                                                               N, zeta, mu, lambdas, eta, Triple)
        list_triple.append(hil.Triple(NN_Options, NN_Actions, NN_Termination))
        trajHIL, controlHIL, optionHIL, terminationHIL, flagHIL = sim.HierarchicalPolicySim(env, list_triple[k], zeta, mu, max_epoch, 
                                                                                            100, option_space, size_input)
        k+=1
        length_traj = np.empty((0))
        for j in range(len(trajHIL)):
            length_traj = np.append(length_traj, len(trajHIL[j][:]))
        averageHIL = np.append(averageHIL,np.divide(np.sum(length_traj),len(length_traj)))
        success_percentageHIL = np.append(success_percentageHIL,np.divide(np.sum(flagHIL),len(length_traj)))
        
Bestid = kb.argmin(averageHIL)  
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)
lambdas = tf.Variable(initial_value=LAMBDAS[Bestid.numpy()]*tf.ones((option_space,)), trainable = False)
eta = tf.Variable(initial_value=ETA[Bestid.numpy()], trainable = False)

# %%

Best_Triple = list_triple[Bestid]
x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'HILvideo', Best_Triple, zeta, mu, max_epoch, option_space, size_input)


fig = plt.figure()
ax1 = plt.subplot(311)
plot_action = plt.scatter(x[:,0], x[:,1], c=o, marker='x', cmap='cool');
cbar = fig.colorbar(plot_action, ticks=[0, 1])
cbar.ax.set_yticklabels(['Option1', 'Option2'])
#plt.xlabel('Position')
plt.ylabel('Velocity')
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(312, sharex=ax1)
plot_action = plt.scatter(x[:,0], x[:,1], c=u, marker='x', cmap='winter');
cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
#plt.xlabel('Position')
plt.ylabel('Velocity')
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(313, sharex=ax1)
plot_action = plt.scatter(x[0:-1,0], x[0:-1,1], c=b, marker='x', cmap='copper');
cbar = fig.colorbar(plot_action, ticks=[0, 1])
cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('Best_Triple_option_state_action_distribution.eps', format='eps')
plt.show()




