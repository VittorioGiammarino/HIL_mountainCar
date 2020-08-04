#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:52:47 2020

@author: vittorio
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import HierarchicalImitationLearning as hil
import numpy as np
import matplotlib.pyplot as plt
import Simulation as sim
import BehavioralCloning as bc
import concurrent.futures
import gym
from joblib import Parallel, delayed
import multiprocessing

# %% Expert Data
bc_data_dir = 'data'
TrainingSet, labels = hil.PreprocessData(bc_data_dir)

TrainingSet = TrainingSet[0:2000,:]
labels = labels[0:2000]
# %% Expert Plot
fig = plt.figure()
plot_action = plt.scatter(TrainingSet[:,0], TrainingSet[:,1], c=labels, marker='x', cmap='winter');
cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('Expert_state_action_distribution.eps', format='eps')
plt.show()


# %% Initialization
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
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

Triple = hil.Triple(NN_options, NN_actions, NN_termination)

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 1000

ED = hil.Experiment_design(labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, 
                           Triple, LAMBDAS, ETA, env, max_epoch)

# %% Regularization 1

lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
NN_Termination, NN_Actions, NN_Options = hil.BaumWelchRegularizer1(ED, lambdas)
Triple_reg1 = hil.Triple(NN_Options, NN_Actions, NN_Termination)

x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'Regularization1', Triple_reg1, zeta, mu, 200, option_space, size_input)

fig = plt.figure()
ax1 = plt.subplot(311)
plot_action = plt.scatter(x[:,0], x[:,1], c=o[:], marker='x', cmap='cool');
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
plt.savefig('Plot_regularization1.eps', format='eps')
plt.show()



# %% Regularization 2
eta = tf.Variable(initial_value=1., trainable=False)

NN_Termination, NN_Actions, NN_Options = hil.BaumWelchRegularizer2(ED, eta)
Triple_reg2 = hil.Triple(NN_Options, NN_Actions, NN_Termination)

x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'Regularization2', Triple_reg1, zeta, mu, 200, option_space, size_input)

fig = plt.figure()
ax1 = plt.subplot(311)
plot_action = plt.scatter(x[:,0], x[:,1], c=o[:], marker='x', cmap='cool');
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
plt.savefig('Plot_regularization2.eps', format='eps')
plt.show()



# %%
inputs = range(len(ETA))
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores, prefer="threads")(delayed(hil.ValidationBW_reward)(i, ED) for i in inputs)

# %%
averageHIL = np.empty((0))
for j in range(len(results)):
    averageHIL = np.append(averageHIL, results[j][1])
    
Bestid = np.argmin(averageHIL) 
Best_Triple = results[Bestid][0]
x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'HILvideo', Best_Triple, zeta, mu, max_epoch, option_space, size_input)

#eta = 1000
#lambda = 3.16

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
plt.savefig('Best_Triple_option_state_action_distribution_Final.eps', format='eps')
plt.show()




