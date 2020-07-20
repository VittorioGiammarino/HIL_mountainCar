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
import Environment as env
import StateSpace as ss
import DynamicProgramming as dp
import Simulation as sim
import BehavioralCloning as bc
import HierachicalImitationLearning as hil
import concurrent.futures

# %% map generation 
map = env.Generate_world_subgoals_simplified()

# %% Generate State Space
stateSpace=ss.GenerateStateSpace(map)            
K = stateSpace.shape[0];
TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
P = dp.ComputeTransitionProbabilityMatrix(stateSpace,map)
G = dp.ComputeStageCosts(stateSpace,map)
[J_opt_vi,u_opt_ind_vi] = dp.ValueIteration(P,G,TERMINAL_STATE_INDEX)

# %% Plot Optimal Solution
env.PlotOptimalSolution(map,stateSpace,u_opt_ind_vi, 'Expert_pickup.eps', 'Expert_dropoff.eps')

# %% Generate Expert's trajectories
T_train=10
base=ss.BaseStateIndex(stateSpace,map)
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T_train, base, TERMINAL_STATE_INDEX)
labels_train, TrainingSet_train = bc.ProcessData(traj,control,stateSpace)
T_validation = 10
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T_validation, base, TERMINAL_STATE_INDEX)
labels_validation, TrainingSet_validation = bc.ProcessData(traj,control,stateSpace)
# %% Simulation
#env.VideoSimulation(map,stateSpace,control[1][:],traj[1][:])

# %% Behavioral Cloning vs Baum-Welch algorithm
option_space = 2
action_space = 5
termination_space = 2

N=10 #Iterations
zeta = 0.1 #Failure factor
mu = np.ones(option_space)*np.divide(1,option_space) #initial option probability distribution

gain_lambdas = np.logspace(1, 3, 7, dtype = 'float32')
gain_eta = np.logspace(1, 3, 7, dtype = 'float32')

list_triple_weights = []
validation = []

NN_options = hil.NN_options(option_space)
NN_actions = hil.NN_actions(action_space)
NN_termination = hil.NN_termination(termination_space)
Triple_weights = hil.Triple_Weights(NN_options.get_weights(), NN_actions.get_weights(), NN_termination.get_weights())


for i in range(len(gain_lambdas)):
    for j in range(len(gain_eta)):
        T = TrainingSet_train.shape[0]
        TrainingSetTermination = hil.TrainingSetTermination(TrainingSet_train, option_space)
        TrainingSetActions, labels_reshaped = hil.TrainingAndLabelsReshaped(option_space,T, TrainingSet_train, labels_train)
        lambdas = tf.Variable(initial_value=gain_lambdas[i]*tf.ones((option_space,)), trainable=False)
        eta = tf.Variable(initial_value=gain_eta[j], trainable=False)
        NN_Termination, NN_Actions, NN_Options = hil.BaumWelch(labels_train, TrainingSet_train, action_space, option_space, termination_space, 
                                                               N, zeta, mu, lambdas, eta, Triple_weights)
        list_triple_weights.append(hil.Triple_Weights(NN_Options.get_weights(), NN_Actions.get_weights(), NN_Termination.get_weights()))
        validation.append(hil.ValidationBW(labels_validation, TrainingSet_validation, action_space, option_space, termination_space, 
                                           zeta, mu, NN_Options, NN_Actions, NN_Termination))
        
Validation = kb.argmin(validation)  
ETA, LAMBDAS = np.meshgrid(gain_eta, gain_lambdas)
LAMBDAS = LAMBDAS.reshape(len(gain_lambdas)*len(gain_eta),)
ETA = ETA.reshape(len(gain_lambdas)*len(gain_eta),)
lambdas = tf.Variable(initial_value=LAMBDAS[Validation.numpy()]*tf.ones((option_space,)), trainable = False)
eta = tf.Variable(initial_value=ETA[Validation.numpy()], trainable = False)

# %%

val = np.empty((len(gain_lambdas),len(gain_eta)))
k=0
for i in range(len(gain_lambdas)):
    for j in range(len(gain_eta)):
        val[j,i] = validation[k].numpy()
        k+=1
    
    
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(gain_eta, gain_lambdas, val, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')
plt.show()


# %%

T_train=100
base=ss.BaseStateIndex(stateSpace,map)
[traj,control,flag]=sim.SampleTrajMDP(P, u_opt_ind_vi, 1000, T_train, base, TERMINAL_STATE_INDEX)
labels_train, TrainingSet_train = bc.ProcessData(traj,control,stateSpace)

ntraj = [1, 5, 10, 20]
average_NN1, success_percentageNN1, average_expert = bc.EvaluationNN1(map, stateSpace, P, traj, control, ntraj)
[averageBW, success_percentageBW,
 list_triple_weights_performance]  = hil.EvaluationBW(map, stateSpace, P, traj, control, ntraj, 
                                                    action_space, option_space, termination_space, 
                                                    N, zeta, mu, lambdas, eta, Triple_weights)

# %% plot of performance 
plt.figure()
plt.subplot(211)
plt.plot(ntraj, average_NN1,'go--', label = 'Neural Network 1')
plt.plot(ntraj, averageBW,'rs--', label = 'Hierarchical Policy')
plt.plot(ntraj, average_expert,'b', label = 'Expert')
plt.ylabel('Average steps to goal')
plt.subplot(212)
plt.plot(ntraj, success_percentageNN1,'go--', label = 'Nerual Network 1')
plt.plot(ntraj, success_percentageBW,'rs--', label = 'Hierarchical Policy')
plt.plot(ntraj, np.ones((len(ntraj))),'b', label='Expert')
plt.xlabel('Number of Trajectories')
plt.ylabel('Percentage of success')
plt.legend(loc='lower right')
plt.savefig('evaluation_BWvsBC.eps', format='eps')
plt.show()

# %%

Best = np.argmin(averageBW)

NN_Options = hil.NN_options(option_space)
NN_Actions = hil.NN_actions(action_space)
NN_Termination = hil.NN_termination(termination_space)
    
NN_Options.set_weights(list_triple_weights_performance[Best].options_weights)
NN_Actions.set_weights(list_triple_weights_performance[Best].actions_weights)
NN_Termination.set_weights(list_triple_weights_performance[Best].termination_weights)

Pi_HI = np.argmax(NN_Options(stateSpace).numpy(),1)    
Pi_Lo_o1 = np.argmax(NN_Actions(hil.TrainingSetPiLo(stateSpace,0)).numpy(),1)
Pi_Lo_o2 = np.argmax(NN_Actions(hil.TrainingSetPiLo(stateSpace,1)).numpy(),1)
# Pi_Lo_o3 = np.argmax(NN_Actions(hil.TrainingSetPiLo(stateSpace,2)).numpy(),1)
Pi_term_1 = (NN_Termination(hil.TrainingSetPiLo(stateSpace,0)).numpy())
Pi_term_2 = (NN_Termination(hil.TrainingSetPiLo(stateSpace,1)).numpy())
# Pi_term_3 = (NN_Termination(hil.TrainingSetPiLo(stateSpace,2)).numpy())

env.PlotOptimalSolution(map,stateSpace,Pi_Lo_o1, 'option1_pickup.eps', 'option1_dropoff.eps')
env.PlotOptimalSolution(map,stateSpace,Pi_Lo_o2, 'option2_pickup.eps', 'option2_dropoff.eps')
# env.PlotOptimalSolution(map,stateSpace,Pi_Lo_o3, 'option3_pickup.eps', 'option3_dropoff.eps')

# %%
Trajs=1
base=ss.BaseStateIndex(stateSpace,map)
[trajHIL,controlHIL,OptionsHIL, 
 TerminationHIL, flagHIL]=sim.HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_Options, 
                                                                  NN_Actions, NN_Termination, mu, 1000, 
                                                                  Trajs, base, TERMINAL_STATE_INDEX, zeta, option_space)
                                                                  
# %%
env.HILVideoSimulation(map,stateSpace,controlHIL[0][:],trajHIL[0][:],OptionsHIL[0][:],"sim_HIL.mp4")
