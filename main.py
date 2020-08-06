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
import pickle

# %% Expert Data
bc_data_dir = 'Expert/Data'
TrainingSet, labels = hil.PreprocessData(bc_data_dir)
TrainingSet = TrainingSet[0:1000,:]
labels = labels[0:1000]

# %% Expert Plot
fig = plt.figure()
plot_action = plt.scatter(TrainingSet[:,0], TrainingSet[:,1], c=labels, marker='x', cmap='winter');
cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('Figures/FiguresExpert/Expert_state_action_distribution.eps', format='eps')
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

gain_lambdas = np.logspace(0, 1.5, 4, dtype = 'float32')
gain_eta = np.logspace(1, 3, 3, dtype = 'float32')
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)

Triple = hil.Triple(NN_options, NN_actions, NN_termination)

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 1000

ED = hil.Experiment_design(labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, 
                           Triple, LAMBDAS, ETA, env, max_epoch)

lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=100., trainable=False)

# %% HMM order estimation

# Model_orders = [1, 2, 3, 4, 5, 6, 7]
# Likelihood = np.empty(0) 
# for d in Model_orders:
#     Likelihood = np.append(Likelihood, -hil.HMM_order_estimation(d, ED))
    
# with open('Variables_saved/likelihood.pickle', 'wb') as f:
#     pickle.dump([Likelihood, Model_orders], f)

# with open('Variables_saved/likelihood.npy', 'wb') as f:
#     np.save(f,[Likelihood, Model_orders])
    
# %% Plot Figure
	
# with open('Variables_saved/likelihood.pickle', 'rb') as f:
#     Likelihood, Model_orders = pickle.load(f)

with open('Variables_saved/likelihood.npy', 'rb') as f:
    Likelihood, Model_orders = np.load(f)

fig = plt.figure()
plot_action = plt.plot(Model_orders, Likelihood,'o--');
plt.xlabel('Model Order')
plt.ylabel('Lower bound for the Likelihood')
plt.savefig('Figures/FiguresHIL/Likelihood_over_order.eps', format='eps')
plt.show()

# %% Regularization 1

lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
NN_Termination, NN_Actions, NN_Options = hil.BaumWelchRegularizer1(ED, lambdas)
Triple_reg1 = hil.Triple(NN_Options, NN_Actions, NN_Termination)

x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'Videos/VideosHIL/Reg1', Triple_reg1, zeta, mu, 200, option_space, size_input)

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
plt.savefig('Figures/FiguresHIL/Reg1/Plot_regularization1.eps', format='eps')
plt.show()



# %% Regularization 2
eta = tf.Variable(initial_value=1., trainable=False)

NN_Termination, NN_Actions, NN_Options = hil.BaumWelchRegularizer2(ED, eta)
Triple_reg2 = hil.Triple(NN_Options, NN_Actions, NN_Termination)

x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'Videos/VideosHIL/Reg2', Triple_reg2, zeta, mu, 200, option_space, size_input)

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
plt.savefig('Figures/FiguresHIL/Reg2/Plot_regularization2.eps', format='eps')
plt.show()


# %% Regularizers validation

inputs = range(len(ETA))
num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores, prefer="threads")(delayed(hil.ValidationBW_reward)(i, ED) for i in inputs)

# %%
averageHIL = np.empty((0))
for j in range(len(results)):
    averageHIL = np.append(averageHIL, results[j][1])
    
Bestid = np.argmin(averageHIL) 
Best_Triple = results[Bestid][0]

Best_Triple.save(ED.gain_lambdas[Bestid], ED.gain_eta[Bestid])

# %%
x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'Videos/VideosHIL/eta_{}_lambda_{}'.format(ED.gain_eta[Bestid], ED.gain_lambdas[Bestid]), 
                                         Best_Triple, zeta, mu, max_epoch, option_space, size_input)

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
plt.savefig('Figures/FiguresHIL/HIL_simulation_eta_{}_lambda_{}.eps'.format(ED.gain_eta[Bestid], ED.gain_lambdas[Bestid]), format='eps')
plt.show()


# %% Evaluation with a different number of samples

# lambdas = tf.Variable(initial_value=ED.gain_lambdas[Bestid]*tf.ones((option_space,)), trainable=False)
# eta = tf.Variable(initial_value=ED.gain_eta[Bestid], trainable=False)
nSamples = [200, 500, 1000, 2000, 3000, 4000]
average_expert = bc.AverageExpert(TrainingSet)
averageBW, success_percentageBW = hil.EvaluationBW(TrainingSet, labels, nSamples, ED, lambdas, eta)

# %%
plt.figure()
plt.subplot(211)
plt.plot(nSamples, averageBW,'go--', label = 'HIL')
plt.plot(nSamples, average_expert*np.ones((len(nSamples))),'b', label = 'Expert')
plt.ylabel('Average steps to goal')
plt.subplot(212)
plt.plot(nSamples, success_percentageBW,'go--', label = 'HIL')
plt.plot(nSamples, np.ones((len(nSamples))),'b', label='Expert')
plt.xlabel('Number of Samples')
plt.ylabel('Percentage of success')
plt.legend(loc='lower right')
plt.savefig('Figures/FiguresHIL/evaluationHIL_multipleTrajs.eps', format='eps')
plt.show()


