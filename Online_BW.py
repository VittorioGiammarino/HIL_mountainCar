#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:17:50 2020

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

# %%
batches_TrainingSet, batches_labels, average = hil.BatchTrainingSet(TrainingSet, labels)

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

N=1 #Iterations
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

