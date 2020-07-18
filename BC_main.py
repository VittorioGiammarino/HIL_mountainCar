#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:02:09 2020

@author: vittorio
"""

import HierarchicalImitationLearning as hil
import BehavioralCloning as bc
import tensorflow as tf
import gym
import numpy as np
import Simulation as sim
import matplotlib.pyplot as plt

# %%    Store Data From Expert.

bc_data_dir = 'data'
TrainingSet, labels = hil.PreprocessData(bc_data_dir)

# %% NN Behavioral Cloning

action_space=2
size_input = TrainingSet.shape[1]
model1 = bc.NN1(action_space, size_input)
model2 = bc.NN2(action_space, size_input)
model3 = bc.NN3(action_space, size_input)

# train the models
model1.fit(TrainingSet, labels, epochs=10)
encoded = tf.keras.utils.to_categorical(labels)
model2.fit(TrainingSet, encoded, epochs=10)
model3.fit(TrainingSet, encoded, epochs=10)

# %%
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 200

traj, control, flag = sim.FlatPolicySim(env, model1, max_epoch, 100, size_input)

# %%

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 200

nSamples = [200, 500, 1000, len(TrainingSet)]
average_NN1, success_percentageNN1 = bc.EvaluationNN1(env, action_space, size_input, labels, TrainingSet, nSamples, max_epoch)
average_NN2, success_percentageNN2 = bc.EvaluationNN2(env, action_space, size_input, labels, TrainingSet, nSamples, max_epoch)
average_NN3, success_percentageNN3 = bc.EvaluationNN3(env, action_space, size_input, labels, TrainingSet, nSamples, max_epoch)
average_expert = bc.AverageExpert(TrainingSet)

    
    
# %% plot performance 
plt.figure()
plt.subplot(211)
plt.plot(nSamples, average_NN1,'go--', label = 'Neural Network 1')
plt.plot(nSamples, average_NN2,'rs--', label = 'Neural Network 2')
plt.plot(nSamples, average_NN3,'cp--', label = 'Neural Network 3')
plt.plot(nSamples, average_expert*np.ones((len(nSamples))),'b', label = 'Expert')
plt.ylabel('Average steps to goal')
plt.subplot(212)
plt.plot(nSamples, success_percentageNN1,'go--', label = 'Nerual Network 1')
plt.plot(nSamples, success_percentageNN2,'rs--', label = 'Nerual Network 2')
plt.plot(nSamples, success_percentageNN3,'cp--', label = 'Nerual Network 3')
plt.plot(nSamples, np.ones((len(nSamples))),'b', label='Expert')
plt.xlabel('Number of Trajectories')
plt.ylabel('Percentage of success')
plt.legend(loc='lower right')
plt.savefig('evaluation.eps', format='eps')
plt.show()
