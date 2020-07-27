#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:36:46 2020

@author: vittorio
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as mpl
import Simulation as sim
import HierarchicalImitationLearning as hil
import gym
import BehavioralCloning as bc
import concurrent.futures
import pickle

# %% map generation 
bc_data_dir = 'data'
TrainingSet, labels = hil.PreprocessData(bc_data_dir)

TrainingSet = TrainingSet[0:4000,:]
labels = labels[0:4000]
# %%
fig = plt.figure()
plot_action = plt.scatter(TrainingSet[:,0], TrainingSet[:,1], c=labels, marker='x', cmap='winter');
cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('Expert_state_action_distribution.eps', format='eps')
plt.show()


# %% Triple parameterization
option_space = 2
action_space = 2
termination_space = 2
size_input = TrainingSet.shape[1]

NN_options = hil.NN_options(option_space, size_input)
NN_actions = hil.NN_actions(action_space, size_input)
NN_termination = hil.NN_termination(termination_space, size_input)

# %% Baum-Welch for provable HIL iteration

N = 10
zeta = 0.1
mu = np.ones(option_space)*np.divide(1,option_space)
T = TrainingSet.shape[0]
TrainingSetTermination = hil.TrainingSetTermination(TrainingSet, option_space, size_input)
TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels, size_input)
lambdas = tf.Variable(initial_value=1.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=100., trainable=False)

for n in range(N):
    print('iter', n, '/', N)
    
    # Uncomment for sequential Running
    # alpha = hil.Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination)
    # beta = hil.Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination)
    # gamma = hil.Gamma(TrainingSet, option_space, termination_space, alpha, beta)
    # gamma_tilde = hil.GammaTilde(TrainingSet, labels, beta, alpha, 
    #                               NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)
    
    
    # MultiThreading Running
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(hil.Alpha, TrainingSet, labels, option_space, termination_space, mu, 
                              zeta, NN_options, NN_actions, NN_termination)
        f2 = executor.submit(hil.Beta, TrainingSet, labels, option_space, termination_space, zeta, 
                              NN_options, NN_actions, NN_termination)  
        alpha = f1.result()
        beta = f2.result()
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f3 = executor.submit(hil.Gamma, TrainingSet, option_space, termination_space, alpha, beta)
        f4 = executor.submit(hil.GammaTilde, TrainingSet, labels, beta, alpha, 
                              NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)  
        gamma = f3.result()
        gamma_tilde = f4.result()
        
    print('Expectation done')
    print('Starting maximization step')
    optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
    epochs = 20 #number of iterations for the maximization step
            
    gamma_tilde_reshaped = hil.GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = hil.GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = hil.GammaReshapeOptions(T, option_space, gamma)
    
    
    # loss = hil.OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
    #                                          TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
    #                                          TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
    #                                          gamma, option_space, labels, size_input)
    
    loss = hil.OptimizeLossAndRegularizerTotBatch(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                                                  TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                                                  TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                                                  gamma, option_space, labels, size_input, 32)

    print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

# %%
Triple = hil.Triple(NN_options, NN_actions, NN_termination)
Triple.save()

# %%

NN_Options, NN_Actions, NN_Termination = hil.Triple.load()
New_Triple = hil.Triple(NN_Options, NN_Actions, NN_Termination)


# %% Evaluation 
Triple = hil.Triple(NN_options, NN_actions, NN_termination)
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 1000

trajHIL, controlHIL, optionHIL, terminationHIL, flagHIL = sim.HierarchicalPolicySim(env, Triple, zeta, mu, max_epoch, 100, option_space, size_input)

positionHIL = np.empty((0))
velocityHIL = np.empty((0))
actionHIL = np.empty((0))
optionsHIL = np.empty((0))
terminationsHIL = np.empty((0))
for j in range(len(trajHIL)):
    for i in range(len(trajHIL[j])-1):
        positionHIL = np.append(positionHIL, trajHIL[j][i][0])
        velocityHIL = np.append(velocityHIL, trajHIL[j][i][1])
        actionHIL = np.append(actionHIL, controlHIL[j][i])
        optionsHIL = np.append(optionsHIL, optionHIL[j][i])
        terminationsHIL = np.append(terminationsHIL, terminationHIL[j][i])
 
length_traj = np.empty((0))
for j in range(len(trajHIL)):
    length_traj = np.append(length_traj, len(trajHIL[j][:]))
averageHIL = np.divide(np.sum(length_traj),len(length_traj))
success_percentageHIL = np.divide(np.sum(flagHIL),len(length_traj))


# %%
fig = plt.figure()
plt.subplot(211)
plot_action = plt.scatter(positionHIL, velocityHIL, c=optionsHIL, marker='x', cmap='cool');
cbar = fig.colorbar(plot_action, ticks=[0, 1])
cbar.ax.set_yticklabels(['Option1', 'Option2'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.subplot(212)
plot_action = plt.scatter(positionHIL, velocityHIL, c=actionHIL, marker='x', cmap='winter');
cbar = fig.colorbar(plot_action, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.savefig('HIL_option_state_action_distribution.eps', format='eps')
plt.show()

# %%

x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'HILvideo', New_Triple, zeta, mu, max_epoch, option_space, size_input)

# %%

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
plt.savefig('HIL_option_state_action_distribution.eps', format='eps')
plt.show()


# %% Preparing Animation

Writer = anim.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=3800)

fig = plt.figure()
ax1 = plt.subplot(311)
ticks = [0, 1]
plot_action = plt.scatter(x[0:2,0], x[0:2,1], c=ticks[0:2], marker='x', cmap='cool');
cbar = fig.colorbar(plot_action)
cbar.set_ticks(ticks)
cbar.set_ticklabels(['Option1', 'Option2'])
#plt.xlabel('Position')
plt.ylabel('Velocity')
ax1.set_xlim(-1.25, 0.55)
ax1.set_ylim(-0.07, 0.07)
plt.setp(ax1.get_xticklabels(), visible=False)
ax2 = plt.subplot(312, sharex=ax1)
plot_option = plt.scatter(x[0:2,0], x[0:2,1], c=ticks[0:2], marker='x', cmap='winter');
cbar = fig.colorbar(plot_option, ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['Left', 'No Action', 'Right'])
#plt.xlabel('Position')
plt.ylabel('Velocity')
ax2.set_ylim(-0.07, 0.07)
plt.setp(ax2.get_xticklabels(), visible=False)
ax3 = plt.subplot(313, sharex=ax1)
plot_termination = plt.scatter(x[0:2,0], x[0:2,1], c=ticks[0:2], marker='x', cmap='copper');
cbar = fig.colorbar(plot_termination, ticks=[0, 1])
cbar.ax.set_yticklabels(['Same Option', 'Terminate'])
plt.xlabel('Position')
plt.ylabel('Velocity')
ax3.set_ylim(-0.07, 0.07)
#plt.savefig('HIL_option_state_action_distribution.eps', format='eps')
plt.show()

#%% Animation


def animation_frame(i, x, o, u, b):
    plot_action.set_offsets(x[0:i,:])
    plot_action.set_sizes(10*np.ones(i))
    plot_action.set_array(o[0:i])
    plot_option.set_offsets(x[0:i,:])
    plot_option.set_sizes(10*np.ones(i))
    plot_option.set_array(u[0:i])
    plot_termination.set_offsets(x[0:i,:])
    plot_termination.set_sizes(10*np.ones(i))
    plot_termination.set_array(b[0:i])
    return plot_action, plot_option, plot_termination,


animation = anim.FuncAnimation(fig, func = animation_frame, frames=b.shape[0], fargs=(x, o, u, b))
animation.save('animation.mp4', writer=writer)


