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

bc_data_dir = 'Expert/Data'
TrainingSet, labels = hil.PreprocessData(bc_data_dir)

# %% NN Behavioral Cloning

action_space=2
size_input = TrainingSet.shape[1]
model1 = bc.NN1(action_space, size_input)
model2 = bc.NN2(action_space, size_input)
model3 = bc.NN3(action_space, size_input)

# train the models
model1.fit(TrainingSet, labels, epochs=50)
encoded = tf.keras.utils.to_categorical(labels)
model2.fit(TrainingSet, encoded, epochs=50)
model3.fit(TrainingSet, encoded, epochs=50)

# %%

env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 200

nSamples = [200, 500, 1000, 2000, 3000, len(TrainingSet)]
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
plt.xlabel('Number of Samples')
plt.ylabel('Percentage of success')
plt.legend(loc='lower right')
plt.savefig('Figures/FiguresBC/evaluationBehavioralCloning.eps', format='eps')
plt.show()

# %% 
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 200

traj, control, flag = sim.FlatPolicySim(env, model2, max_epoch, 100, size_input)

position = np.empty((0))
velocity = np.empty((0))
action = np.empty((0))
for j in range(len(traj)):
    for i in range(len(traj[j])):
        position = np.append(position, traj[j][i][0])
        velocity = np.append(velocity, traj[j][i][1])
        action = np.append(action, control[j][i])

# %%

fig = plt.figure()
plot_action = plt.scatter(position, velocity, c=action, marker='x', cmap='winter');
cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('Figures/FiguresBC/BC_state_action_distribution.eps', format='eps')
plt.show()
        

# %% Make video sample
sim.VideoFlatPolicy('MountainCar-v0', 'Videos/VideosBC', model3, size_input)

            
