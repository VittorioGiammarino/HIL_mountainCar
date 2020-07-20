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
import Simulation as sim
import gym
import BehavioralCloning as bc
import HierarchicalImitationLearning as hil
import concurrent.futures

# %% map generation 
bc_data_dir = 'data'
TrainingSet, labels = hil.PreprocessData(bc_data_dir)

TrainingSet = TrainingSet[0:1000,:]
labels = labels[0:1000]
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
eta = tf.Variable(initial_value=1., trainable=False)

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
    epochs = 50 #number of iterations for the maximization step
            
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

# %% Evaluation 
Triple = hil.Triple(NN_options, NN_actions, NN_termination)
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1200
max_epoch = 1000

trajHIL, controlHIL, optionHIL, terminationHIL, flagHIL = sim.HierarchicalPolicySim(env, Triple, zeta, mu, max_epoch, 1, option_space, size_input)

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

x, u, o, b = sim.VideoHierarchicalPolicy('MountainCar-v0', 'HILvideo', Triple, zeta, mu, max_epoch, option_space, size_input)


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


# %% Understanding Regularization

option_space = 3
action_space = 5
termination_space = 2

NN_options = hil.NN_options(option_space)
NN_actions = hil.NN_actions(action_space)
NN_termination = hil.NN_termination(termination_space)

ntraj = 10
N = 5
zeta = 0.1
mu = np.ones(option_space)*np.divide(1,option_space)
T = TrainingSet.shape[0]
TrainingSetTermination = hil.TrainingSetTermination(TrainingSet, option_space)
TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels)
lambdas = tf.Variable(initial_value=10.*tf.ones((option_space,)), trainable=False)
eta = tf.Variable(initial_value=100., trainable=False)
chi = tf.Variable(initial_value=0.1, trainable=False)

for n in range(N):
    print('iter', n, '/', N)

    alpha = hil.Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination)
    beta = hil.Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination)
    gamma = hil.Gamma(TrainingSet, option_space, termination_space, alpha, beta)
    gamma_tilde = hil.GammaTilde(TrainingSet, labels, beta, alpha, 
                                  NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)

    optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
    epochs = 50 #number of iterations for the maximization step
    
    gamma_tilde_reshaped = hil.GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = hil.GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = hil.GammaReshapeOptions(T, option_space, gamma)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
            tape.watch(weights)
            # Regularization 1
            regular_loss = 0
            for i in range(option_space):
                option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
                option_concat = kb.concatenate((option,option),1)
                log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
                policy_termination = NN_termination(hil.TrainingSetPiLo(TrainingSet,i))
                array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
                for j in range(T):
                    array = array.write(j,NN_actions(hil.TrainingSetPiLo(TrainingSet,i))[j,kb.cast(labels[j],'int32')])
                policy_action = array.stack()
                policy_action_reshaped = kb.reshape(policy_action,(T,1))
                policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
                
                regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
        
            # Regularization 2
            ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            for i in range(option_space):
                ta = ta.write(i,kb.sum(-kb.sum(NN_actions(hil.TrainingSetPiLo(TrainingSet,i))*kb.log(
                                NN_actions(hil.TrainingSetPiLo(TrainingSet,i))),1)/T,0))
            responsibilities = ta.stack()
    
            values = kb.sum(lambdas*responsibilities) 
            
            # Regularization 3
            ta_op = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            ta_op = ta_op.write(0,-kb.sum(NN_options(TrainingSet)*kb.log(NN_options(TrainingSet)))/T)
            resp_options = ta_op.stack()
    
            entro_options = chi*resp_options 
            
            pi_b = NN_termination(TrainingSetTermination,training=True)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            pi_hi = NN_options(TrainingSet,training=True)
            
            loss_termination = kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
            loss_options = kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
            loss_action = (kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
        
            loss = -values #eta*regular_loss #-entro_options #-loss_termination - loss_action -loss_options -entro_options -values

            
        grads = tape.gradient(loss,weights)
        #optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        #optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        print('options loss:', float(loss))



    
    



