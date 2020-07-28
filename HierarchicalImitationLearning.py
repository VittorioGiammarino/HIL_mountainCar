#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:57:36 2020

@author: vittorio
"""

import tensorflow as tf 
import numpy as np
import argparse
import os
import Simulation as sim
from tensorflow import keras
import tensorflow.keras.backend as kb
import BehavioralCloning as bc
import concurrent.futures

def PreprocessData(bc_data_dir):

    states, actions = [], []
    shards = [x for x in os.listdir(bc_data_dir) if x.endswith('.npy')]
    print("Processing shards: {}".format(shards))

    for shard in shards:
        shard_path = os.path.join(bc_data_dir, shard)
        with open(shard_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            shard_states, unprocessed_actions = zip(*data)
            shard_states = [x.flatten() for x in shard_states]
            
            # Add the shard to the dataset
            states.extend(shard_states)
            actions.extend(unprocessed_actions)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)/2
    
    return states, actions


def NN_options(option_space,size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input,)),
    keras.layers.Dense(option_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_options.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_actions(action_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input+1,)),
    keras.layers.Dense(action_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_actions.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_termination(termination_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input+1,)),
    keras.layers.Dense(termination_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresHIL/model_NN_termination.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def Pi_hi(ot, Pi_hi_parameterization, state):

    Pi_hi = Pi_hi_parameterization(state)
    o_prob = Pi_hi[0,ot]
    
    return o_prob

def Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space):
    if b == True:
        o_prob_tilde = Pi_hi(ot, Pi_hi_parameterization, state)
    elif ot == ot_past:
        o_prob_tilde = 1-zeta+np.divide(zeta,option_space)
    else:
        o_prob_tilde = np.divide(zeta,option_space)
        
    return o_prob_tilde

def Pi_lo(a, Pi_lo_parameterization, state_and_option):
    Pi_lo = Pi_lo_parameterization(state_and_option)
    a_prob = Pi_lo[0,int(a)]
    
    return a_prob

def Pi_b(b, Pi_b_parameterization, state_and_option):
    Pi_b = Pi_b_parameterization(state_and_option)
    if b == True:
        b_prob = Pi_b[0,1]
    else:
        b_prob = Pi_b[0,0]
        
    return b_prob
    
def Pi_combined(ot, ot_past, a, b, Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, state, zeta, option_space):
    Pi_hi_eval = Pi_hi_bar(b, ot, ot_past, Pi_hi_parameterization, state, zeta, option_space)
    Pi_lo_eval = Pi_lo(a, Pi_lo_parameterization, np.append(state, [[ot]],axis=1))
    Pi_b_eval = Pi_b(b, Pi_b_parameterization, np.append(state, [[ot]],axis=1))
    output = Pi_hi_eval*Pi_lo_eval*Pi_b_eval
    
    return output
    
def ForwardRecursion(alpha_past, a, Pi_hi_parameterization, Pi_lo_parameterization,
                     Pi_b_parameterization, state, zeta, option_space, termination_space):
# =============================================================================
#     alpha is the forward message: alpha.shape()= [option_space, termination_space]
# =============================================================================
    alpha = np.empty((option_space, termination_space))
    for i1 in range(option_space):
        ot = i1
        for i2 in range(termination_space):
            if i2 == 1:
                bt=True
            else:
                bt=False
            
            Pi_comb = np.zeros(option_space)
            for ot_past in range(option_space):
                Pi_comb[ot_past] = Pi_combined(ot, ot_past, a, bt, 
                                               Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, 
                                               state, zeta, option_space)
            alpha[ot,i2] = np.dot(alpha_past[:,0],Pi_comb)+np.dot(alpha_past[:,1],Pi_comb)  
    alpha = np.divide(alpha,np.sum(alpha))
            
    return alpha
    
def ForwardFirstRecursion(mu, a, Pi_hi_parameterization, Pi_lo_parameterization,
                          Pi_b_parameterization, state, zeta, option_space, termination_space):
# =============================================================================
#     alpha is the forward message: alpha.shape()=[option_space, termination_space]
#   mu is the initial distribution over options: mu.shape()=[1,option_space]
# =============================================================================
    alpha = np.empty((option_space, termination_space))
    for i1 in range(option_space):
        ot = i1
        for i2 in range(termination_space):
            if i2 == 1:
                bt=True
            else:
                bt=False
            
            Pi_comb = np.zeros(option_space)
            for ot_past in range(option_space):
                Pi_comb[ot_past] = Pi_combined(ot, ot_past, a, bt, 
                                               Pi_hi_parameterization, Pi_lo_parameterization, Pi_b_parameterization, 
                                               state, zeta, option_space)
            alpha[ot,i2] = np.dot(mu, Pi_comb[:])    
    alpha = np.divide(alpha, np.sum(alpha))
            
    return alpha

def BackwardRecursion(beta_next, a, Pi_hi_parameterization, Pi_lo_parameterization,
                      Pi_b_parameterization, state, zeta, option_space, termination_space):
# =============================================================================
#     beta is the backward message: beta.shape()= [option_space, termination_space]
# =============================================================================
    beta = np.empty((option_space, termination_space))
    for i1 in range(option_space):
        ot = i1
        for i2 in range(termination_space):
            for i1_next in range(option_space):
                ot_next = i1_next
                for i2_next in range(termination_space):
                    if i2 == 1:
                        b_next=True
                    else:
                        b_next=False
                    beta[i1,i2] = beta[i1,i2] + beta_next[ot_next,i2_next]*Pi_combined(ot_next, ot, a, b_next, 
                                                                                       Pi_hi_parameterization, Pi_lo_parameterization, 
                                                                                       Pi_b_parameterization, state, zeta, option_space)
    beta = np.divide(beta,np.sum(beta))
    
    return beta

def Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_options, NN_actions, NN_termination):
    alpha = np.empty((option_space,termination_space,len(TrainingSet)))
    for t in range(len(TrainingSet)):
        print('alpha iter', t+1, '/', len(TrainingSet))
        if t ==0:
            state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
            action = labels[t]
            alpha[:,:,t] = ForwardFirstRecursion(mu, action, NN_options, 
                                                 NN_actions, NN_termination, 
                                                 state, zeta, option_space, termination_space)
        else:
            state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
            action = labels[t]
            alpha[:,:,t] = ForwardRecursion(alpha[:,:,t-1], action, NN_options, 
                                            NN_actions, NN_termination, 
                                            state, zeta, option_space, termination_space)
           
    return alpha

def Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_options, NN_actions, NN_termination):
    beta = np.empty((option_space,termination_space,len(TrainingSet)))
    beta[:,:,len(TrainingSet)-1] = np.divide(np.ones((option_space,termination_space)),2*option_space)
    
    for t_raw in range(len(TrainingSet)-1):
        t = len(TrainingSet) - (t_raw+1)
        print('beta iter', t_raw+1, '/', len(TrainingSet)-1)
        state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
        action = labels[t]
        beta[:,:,t-1] = BackwardRecursion(beta[:,:,t], action, NN_options, 
                                        NN_actions, NN_termination, state, zeta, 
                                        option_space, termination_space)
        
    return beta

def Smoothing(option_space, termination_space, alpha, beta):
    gamma = np.empty((option_space, termination_space))
    for i1 in range(option_space):
        ot=i1
        for i2 in range(termination_space):
            gamma[ot,i2] = alpha[ot,i2]*beta[ot,i2]     
    gamma = np.divide(gamma,np.sum(gamma))
    
    return gamma

def DoubleSmoothing(beta, alpha, a, Pi_hi_parameterization, Pi_lo_parameterization, 
                    Pi_b_parameterization, state, zeta, option_space, termination_space):
    gamma_tilde = np.empty((option_space, termination_space))
    for i1_past in range(option_space):
        ot_past = i1_past
        for i2 in range(termination_space):
            if i2 == 1:
                b=True
            else:
                b=False
            for i1 in range(option_space):
                ot = i1
                gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2] + beta[ot,i2]*Pi_combined(ot, ot_past, a, b, 
                                                                                  Pi_hi_parameterization, Pi_lo_parameterization, 
                                                                                  Pi_b_parameterization, state, zeta, option_space)
            gamma_tilde[ot_past,i2] = gamma_tilde[ot_past,i2]*np.sum(alpha[ot_past,:])
    gamma_tilde = np.divide(gamma_tilde,np.sum(gamma_tilde))
    
    return gamma_tilde

def Gamma(TrainingSet, option_space, termination_space, alpha, beta):
    gamma = np.empty((option_space,termination_space,len(TrainingSet)))
    for t in range(len(TrainingSet)):
        print('gamma iter', t+1, '/', len(TrainingSet))
        gamma[:,:,t]=Smoothing(option_space, termination_space, alpha[:,:,t], beta[:,:,t])
        
    return gamma

def GammaTilde(TrainingSet, labels, beta, alpha, Pi_hi_parameterization, Pi_lo_parameterization, 
               Pi_b_parameterization, zeta, option_space, termination_space):
    gamma_tilde = np.empty((option_space,termination_space,len(TrainingSet)))
    for t in range(1,len(TrainingSet)):
        print('gamma tilde iter', t, '/', len(TrainingSet)-1)
        state = TrainingSet[t,:].reshape(1,len(TrainingSet[t,:]))
        action = labels[t]
        gamma_tilde[:,:,t]=DoubleSmoothing(beta[:,:,t], alpha[:,:,t-1], action, 
                                           Pi_hi_parameterization, Pi_lo_parameterization, 
                                           Pi_b_parameterization, state, zeta, option_space, termination_space)
        
    return gamma_tilde
    
def TrainingSetTermination(TrainingSet,option_space, size_input):
    # Processing termination
    T = len(TrainingSet)
    TrainingSet_reshaped_termination = np.empty((int(option_space*(T-1)),size_input+1))
    j=1
    for i in range(0,option_space*(T-1),option_space):
        for k in range(option_space):
            TrainingSet_reshaped_termination[i+k,:] = np.append(TrainingSet[j,:], [[k]])
        j+=1
        
    return TrainingSet_reshaped_termination
        
def GammaTildeReshape(gamma_tilde, option_space):
    T = gamma_tilde.shape[2]
    gamma_tilde_reshaped = np.empty((int(option_space*(T-1)),2),dtype='float32')
    j=1
    for i in range(0,option_space*(T-1),option_space):
        gamma_tilde_reshaped[i:i+option_space,:] = gamma_tilde[:,:,j]
        j+=1
        
    return gamma_tilde_reshaped

def OptimizeNNtermination(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, T, optimizer):
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_termination.trainable_weights)
            pi_b = NN_termination(TrainingSetTermination,training=True)
            loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_termination, NN_termination.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_termination.trainable_weights))
        print('termination loss:', float(loss_termination))
        
    return loss_termination
    
        
def TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels, size_input):
    TrainingSet_reshaped_actions = np.empty((int(option_space*(T)),size_input+1))
    labels_reshaped = np.empty((int(option_space*(T)),1))
    j=0
    for i in range(0,option_space*(T),option_space):
        for k in range(option_space):
            TrainingSet_reshaped_actions[i+k,:] = np.append(TrainingSet[j,:], [[k]])
            labels_reshaped[i+k,:] = labels[j]
        j+=1
        
    return TrainingSet_reshaped_actions, labels_reshaped

def GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped):
    gamma_reshaped = np.empty((int(option_space*(T)),2),dtype='float32')
    j=0
    for i in range(0,option_space*(T),option_space):
        gamma_reshaped[i:i+option_space,:] = gamma[:,:,j]
        j+=1
    
    gamma_actions_false = np.empty((int(option_space*T),action_space))
    for i in range(option_space*T):
        for j in range(action_space):
            if int(labels_reshaped[i])==j:
                gamma_actions_false[i,j]=gamma_reshaped[i,0]
            else:
                gamma_actions_false[i,j] = 0
            
    gamma_actions_true = np.empty((int(option_space*T),action_space))
    for i in range(option_space*T):
        for j in range(action_space):
            if int(labels_reshaped[i])==j:
                gamma_actions_true[i,j]=gamma_reshaped[i,1]
            else:
                gamma_actions_true[i,j] = 0   
                
    return gamma_actions_false, gamma_actions_true

def OptimizeNNactions(epochs, TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true, T, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_actions.trainable_weights)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_action, NN_actions.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_actions.trainable_weights))
        print('action loss:', float(loss_action))
        
    return loss_action
        
def GammaReshapeOptions(T, option_space, gamma):
    gamma_reshaped_options = np.empty((T,option_space),dtype='float32')
    for i in range(T):
        gamma_reshaped_options[i,:] = gamma[:,1,i]
        
    return gamma_reshaped_options

def OptimizeNNoptions(epochs, TrainingSet, NN_options, gamma_reshaped_options, T, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # # Iterate over the batches of the dataset.
        # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            tape.watch(NN_options.trainable_weights)
            pi_hi = NN_options(TrainingSet,training=True)
            loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_options, NN_options.trainable_weights)
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, NN_options.trainable_weights))
        print('options loss:', float(loss_options))
        
    return loss_options

def TrainingSetPiLo(TrainingSet,o, size_input):
    # Processing termination
    T = len(TrainingSet)
    TrainingSet_PiLo = np.empty((T,size_input+1))
    for i in range(T):
        TrainingSet_PiLo[i,:] = np.append(TrainingSet[i,:], [[o]])
        
    return TrainingSet_PiLo    


def RegularizedLoss1(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    pi_b, pi_hi, pi_lo, responsibilities, lambdas, T):
    
    values = -kb.sum(lambdas*responsibilities)
    loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    loss = loss_termination+loss_options+loss_action+values
    
    return loss
    

def OptimizeLossAndRegularizer1(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, lambdas, T, optimizer, option_space):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights, lambdas]
            tape.watch(weights)
            for i in range(option_space):
                ta.write(i,kb.sum(-kb.sum(NN_actions(TrainingSetPiLo(TrainingSet,i))*kb.log(
                        NN_actions(TrainingSetPiLo(TrainingSet,i))),1)/T,0))
            responsibilities = ta.stack()
            pi_b = NN_termination(TrainingSetTermination,training=True)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            pi_hi = NN_options(TrainingSet,training=True)
            loss = RegularizedLoss1(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                                       pi_b, pi_hi, pi_lo, responsibilities, lambdas, T)
            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        optimizer.apply_gradients([(grads[3][:],lambdas)])
        print('options loss:', float(loss))
        
    return loss

            
def Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    pi_b, pi_hi, pi_lo,T):
    
    loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    loss = loss_termination+loss_options+loss_action
    
    return loss    

    
def OptimizeLoss(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                 TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                 TrainingSet, NN_options, gamma_reshaped_options, T, optimizer):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
            tape.watch(weights)
            pi_b = NN_termination(TrainingSetTermination,training=True)
            pi_lo = NN_actions(TrainingSetActions,training=True)
            pi_hi = NN_options(TrainingSet,training=True)
            loss = Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                        pi_b, pi_hi, pi_lo, T)
            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        print('options loss:', float(loss))
        
    return loss
    
def RegularizedLoss2(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, gamma, T, option_space, labels):
    
    pi_b = NN_termination(TrainingSetTermination,training=True)
    pi_lo = NN_actions(TrainingSetActions,training=True)
    pi_hi = NN_options(TrainingSet,training=True)
    regular_loss = 0
    for i in range(option_space):
        option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
        option_concat = kb.concatenate((option,option),1)
        log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
        policy_termination = NN_termination(TrainingSetPiLo(TrainingSet,i))
        array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for j in range(T):
            array = array.write(j,NN_actions(TrainingSetPiLo(TrainingSet,i))[j,kb.cast(labels[j],'int32')])
        policy_action = array.stack()
        policy_action_reshaped = kb.reshape(policy_action,(T,1))
        policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
        regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
    loss_termination = -kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = -kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = -(kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    loss = loss_termination+loss_options+loss_action+eta*regular_loss
    
    return loss
    

def OptimizeLossAndRegularizer2(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, T, optimizer, 
                               gamma, option_space, labels):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights, eta]
            tape.watch(weights)
            loss = RegularizedLoss2(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, gamma, T, option_space, labels)

            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        optimizer.apply_gradients([(grads[3][:],eta)])
        print('options loss:', float(loss))
        
    return loss


def RegularizedLossTot(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, lambdas, gamma, T, option_space, labels, size_input):
    
    pi_b = NN_termination(TrainingSetTermination,training=True)
    pi_lo = NN_actions(TrainingSetActions,training=True)
    pi_hi = NN_options(TrainingSet,training=True)
    
    # Regularization 1
    regular_loss = 0
    for i in range(option_space):
        option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
        option_concat = kb.concatenate((option,option),1)
        log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
        policy_termination = NN_termination(TrainingSetPiLo(TrainingSet,i,size_input))
        array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for j in range(T):
            array = array.write(j,NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))[j,kb.cast(labels[j],'int32')])
        policy_action = array.stack()
        policy_action_reshaped = kb.reshape(policy_action,(T,1))
        policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
        regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
        
    # Regularization 2
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(option_space):
        ta = ta.write(i,kb.sum(-kb.sum(NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))*kb.log(
                        NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))),1)/T,0))
    responsibilities = ta.stack()
    
    values = kb.sum(lambdas*responsibilities) 
    loss_termination = kb.sum(gamma_tilde_reshaped*kb.log(pi_b))/(T)
    loss_options = kb.sum(gamma_reshaped_options*kb.log(pi_hi))/(T)
    loss_action = (kb.sum(gamma_actions_true*kb.log(pi_lo))+kb.sum(gamma_actions_false*kb.log(pi_lo)))/(T)
    
    loss = -loss_termination - loss_options - loss_action + eta*regular_loss - values
    
    return loss



def OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                               gamma, option_space, labels, size_input):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
            tape.watch(weights)
            loss = RegularizedLossTot(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, lambdas, gamma, T, option_space, labels, size_input)

            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        print('options loss:', float(loss))
        
    return loss

def OptimizeLossAndRegularizerTotBatch(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                               gamma, option_space, labels, size_input, size_batch):
    
    n_batches = np.int(TrainingSet.shape[0]/size_batch)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for n in range(n_batches):
            print("\n Batch %d" % (n+1,))
            with tf.GradientTape() as tape:
                weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
                tape.watch(weights)
                loss = RegularizedLossTot(gamma_tilde_reshaped[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          gamma_reshaped_options[n*size_batch:(n+1)*size_batch,:], 
                                          gamma_actions_true[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          gamma_actions_false[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          NN_termination, NN_options, NN_actions,
                                          TrainingSetTermination[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          TrainingSetActions[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                          TrainingSet[n*size_batch:(n+1)*size_batch,:], 
                                          eta, lambdas, gamma[:,:,n*size_batch:(n+1)*size_batch], 
                                          size_batch, option_space, labels, size_input)

            
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
            optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
            optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
            print('options loss:', float(loss))
            
        
    return loss


def BaumWelch(EV, lambdas, eta):
    NN_Options = EV.Triple_init.NN_options
    NN_Actions = EV.Triple_init.NN_actions
    NN_Termination = EV.Triple_init.NN_termination
        
    T = EV.TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(EV.TrainingSet, EV.option_space, EV.size_input)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(EV.option_space,T, EV.TrainingSet, EV.labels, EV.size_input)

    for n in range(EV.N):
        print('iter', n, '/', EV.N)
        
        alpha = Alpha(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
                      EV.zeta, NN_Options, NN_Actions, NN_Termination)
        beta = Beta(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
                    NN_Options, NN_Actions, NN_Termination)
        gamma = Gamma(EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        gamma_tilde = GammaTilde(EV.TrainingSet, EV.labels, beta, alpha, 
                                 NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)
    
        # MultiThreading Running
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f1 = executor.submit(Alpha, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
        #                          EV.zeta, NN_Options, NN_Actions, NN_Termination)
        #     f2 = executor.submit(Beta, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
        #                          NN_Options, NN_Actions, NN_Termination)  
        #     alpha = f1.result()
        #     beta = f2.result()
        
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f3 = executor.submit(Gamma, EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        #     f4 = executor.submit(GammaTilde, EV.TrainingSet, EV.labels, beta, alpha, 
        #                       NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)  
        #     gamma = f3.result()
        #     gamma_tilde = f4.result()
        
        print('Expectation done')
        print('Starting maximization step')
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 50 #number of iterations for the maximization step
            
        gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, EV.option_space)
        gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, EV.option_space, EV.action_space, gamma, labels_reshaped)
        gamma_reshaped_options = GammaReshapeOptions(T, EV.option_space, gamma)
    
    
        # loss = hil.OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
        #                                          TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
        #                                          TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
        #                                          gamma, option_space, labels, size_input)
    
        loss = OptimizeLossAndRegularizerTotBatch(epochs, TrainingSet_Termination, NN_Termination, gamma_tilde_reshaped, 
                                                  TrainingSet_Actions, NN_Actions, gamma_actions_false, gamma_actions_true,
                                                  EV.TrainingSet, NN_Options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                                                  gamma, EV.option_space, EV.labels, EV.size_input, 32)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
    return NN_Termination, NN_Actions, NN_Options


def ValidationBW_reward(i, Experiment_Vars):
        lambdas = tf.Variable(initial_value=Experiment_Vars.gain_lambdas[i]*tf.ones((Experiment_Vars.option_space,)), trainable=False)
        eta = tf.Variable(initial_value=Experiment_Vars.gain_eta[i], trainable=False)
        NN_Termination, NN_Actions, NN_Options = BaumWelch(Experiment_Vars, lambdas, eta)
        list_triple = Triple(NN_Options, NN_Actions, NN_Termination)
        [trajHIL, controlHIL, optionHIL, 
         terminationHIL, flagHIL] = sim.HierarchicalPolicySim(Experiment_Vars.env, list_triple, 
                                                              Experiment_Vars.zeta, Experiment_Vars.mu, Experiment_Vars.max_epoch, 
                                                              100, Experiment_Vars.option_space, Experiment_Vars.size_input)
        length_traj = np.empty((0))
        for j in range(len(trajHIL)):
            length_traj = np.append(length_traj, len(trajHIL[j][:]))
        averageHIL = np.divide(np.sum(length_traj),len(length_traj))
        success_percentageHIL = np.divide(np.sum(flagHIL),len(length_traj))
        
        return list_triple, averageHIL, success_percentageHIL


def ValidationBW(labels, TrainingSet, action_space, option_space, termination_space, zeta, mu, NN_Options, NN_Actions, NN_Termination):
    T = TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(TrainingSet, option_space)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels)
    
    # Uncomment for sequential Running
    alpha = Alpha(TrainingSet, labels, option_space, termination_space, mu, zeta, NN_Options, NN_Actions, NN_Termination)
    beta = Beta(TrainingSet, labels, option_space, termination_space, zeta, NN_Options, NN_Actions, NN_Termination)
    gamma = Gamma(TrainingSet, option_space, termination_space, alpha, beta)
    gamma_tilde = GammaTilde(TrainingSet, labels, beta, alpha, 
                             NN_Options, NN_Actions, NN_Termination, zeta, option_space, termination_space)
    
    # MultiThreading Running
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f1 = executor.submit(hil.Alpha, TrainingSet, labels, option_space, termination_space, mu, 
        #                           zeta, NN_options, NN_actions, NN_termination)
        #     f2 = executor.submit(hil.Beta, TrainingSet, labels, option_space, termination_space, zeta, 
        #                           NN_options, NN_actions, NN_termination)  
        #     alpha = f1.result()
        #     beta = f2.result()
        
    # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     f3 = executor.submit(hil.Gamma, TrainingSet, option_space, termination_space, alpha, beta)
        #     f4 = executor.submit(hil.GammaTilde, TrainingSet, labels, beta, alpha, 
        #                           NN_options, NN_actions, NN_termination, zeta, option_space, termination_space)  
        #     gamma = f3.result()
        #     gamma_tilde = f4.result()
        
    print('Expectation done')
    print('Starting maximization step')
        
    gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, option_space)
    gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
    gamma_reshaped_options = GammaReshapeOptions(T, option_space, gamma)
    
    pi_b = NN_Termination(TrainingSet_Termination)
    pi_lo = NN_Actions(TrainingSet_Actions)
    pi_hi = NN_Options(TrainingSet)

    loss = Loss(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, pi_b, pi_hi, pi_lo, T)

    print(float(loss))
        
    return loss

def Regularizer1(TrainingSet, option_space, size_input, NN_actions, T, lambdas):
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(option_space):
        ta = ta.write(i,kb.sum(-kb.sum(NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))*kb.log(
                        NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))),1)/T,0))
    responsibilities = ta.stack()
    values = kb.sum(lambdas*responsibilities) 
    
    return -values


def OptimizeRegularizer1Batch(epochs, NN_termination, NN_actions, 
                               TrainingSet, NN_options, lambdas, optimizer, option_space, size_input, size_batch):
    
    n_batches = np.int(TrainingSet.shape[0]/size_batch)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for n in range(n_batches):
            print("\n Batch %d" % (n+1,))
            with tf.GradientTape() as tape:
                weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
                tape.watch(weights)
                loss = Regularizer1(TrainingSet[n*size_batch:(n+1)*size_batch,:], option_space, size_input, 
                                    NN_actions, size_batch, lambdas)
            
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
            print('options loss:', float(loss))
               
    return loss    
  
def BaumWelchRegularizer1(EV, lambdas):
    NN_Options = EV.Triple_init.NN_options
    NN_Actions = EV.Triple_init.NN_actions
    NN_Termination = EV.Triple_init.NN_termination
        
    for n in range(1):
        print('iter', n, '/', EV.N)

        print('Starting maximization step')
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 100 #number of iterations for the maximization step
                
        loss = OptimizeRegularizer1Batch(epochs, NN_Termination, NN_Actions, 
                                         EV.TrainingSet, NN_Options, lambdas, optimizer, EV.option_space, EV.size_input, 32)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
    return NN_Termination, NN_Actions, NN_Options
    
def Regularizer2(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, gamma, T, option_space, labels, size_input):
    # Regularization 1
    regular_loss = 0
    for i in range(option_space):
        option =kb.reshape(NN_options(TrainingSet)[:,i],(T,1))
        option_concat = kb.concatenate((option,option),1)
        log_gamma = kb.cast(kb.transpose(kb.log(gamma[i,:,:])),'float32' )
        policy_termination = NN_termination(TrainingSetPiLo(TrainingSet,i,size_input))
        array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        for j in range(T):
            array = array.write(j,NN_actions(TrainingSetPiLo(TrainingSet,i,size_input))[j,kb.cast(labels[j],'int32')])
        policy_action = array.stack()
        policy_action_reshaped = kb.reshape(policy_action,(T,1))
        policy_action_final = kb.concatenate((policy_action_reshaped,policy_action_reshaped),1)
        regular_loss = regular_loss -kb.sum(policy_action_final*option_concat*policy_termination*log_gamma)/T
        
    return eta*regular_loss
    
def OptimizeRegularizer2Batch(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, T, optimizer, 
                               gamma, option_space, labels, size_input, size_batch):
    
    n_batches = np.int(TrainingSet.shape[0]/size_batch)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for n in range(n_batches):
            print("\n Batch %d" % (n+1,))
            with tf.GradientTape() as tape:
                weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
                tape.watch(weights)
                loss = Regularizer2(gamma_tilde_reshaped[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    gamma_reshaped_options[n*size_batch:(n+1)*size_batch,:], 
                                    gamma_actions_true[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    gamma_actions_false[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    NN_termination, NN_options, NN_actions,
                                    TrainingSetTermination[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    TrainingSetActions[option_space*n*size_batch:option_space*(n+1)*size_batch,:], 
                                    TrainingSet[n*size_batch:(n+1)*size_batch,:], 
                                    eta, gamma[:,:,n*size_batch:(n+1)*size_batch], 
                                    size_batch, option_space, labels, size_input)

            
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
            optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
            optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
            print('options loss:', float(loss))
            
        
    return loss    

def BaumWelchRegularizer2(EV, eta):
    NN_Options = EV.Triple_init.NN_options
    NN_Actions = EV.Triple_init.NN_actions
    NN_Termination = EV.Triple_init.NN_termination
        
    T = EV.TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(EV.TrainingSet, EV.option_space, EV.size_input)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(EV.option_space,T, EV.TrainingSet, EV.labels, EV.size_input)

    for n in range(2):
        print('iter', n, '/', EV.N)
        
        # alpha = Alpha(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
        #               EV.zeta, NN_Options, NN_Actions, NN_Termination)
        # beta = Beta(EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
        #             NN_Options, NN_Actions, NN_Termination)
        # gamma = Gamma(EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
        # gamma_tilde = GammaTilde(EV.TrainingSet, EV.labels, beta, alpha, 
        #                          NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)
    
        # MultiThreading Running
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f1 = executor.submit(Alpha, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.mu, 
                                  EV.zeta, NN_Options, NN_Actions, NN_Termination)
            f2 = executor.submit(Beta, EV.TrainingSet, EV.labels, EV.option_space, EV.termination_space, EV.zeta, 
                                  NN_Options, NN_Actions, NN_Termination)  
            alpha = f1.result()
            beta = f2.result()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            f3 = executor.submit(Gamma, EV.TrainingSet, EV.option_space, EV.termination_space, alpha, beta)
            f4 = executor.submit(GammaTilde, EV.TrainingSet, EV.labels, beta, alpha, 
                              NN_Options, NN_Actions, NN_Termination, EV.zeta, EV.option_space, EV.termination_space)  
            gamma = f3.result()
            gamma_tilde = f4.result()
        
        print('Expectation done')
        print('Starting maximization step')
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 50 #number of iterations for the maximization step
            
        gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, EV.option_space)
        gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, EV.option_space, EV.action_space, gamma, labels_reshaped)
        gamma_reshaped_options = GammaReshapeOptions(T, EV.option_space, gamma)
    
    
        # loss = hil.OptimizeLossAndRegularizerTot(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
        #                                          TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
        #                                          TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
        #                                          gamma, option_space, labels, size_input)
    
        loss = OptimizeRegularizer2Batch(epochs, TrainingSet_Termination, NN_Termination, gamma_tilde_reshaped, 
                                                  TrainingSet_Actions, NN_Actions, gamma_actions_false, gamma_actions_true,
                                                  EV.TrainingSet, NN_Options, gamma_reshaped_options, eta, T, optimizer, 
                                                  gamma, EV.option_space, EV.labels, EV.size_input, 32)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))

        
    return NN_Termination, NN_Actions, NN_Options
    
class Triple:
    def __init__(self, NN_options, NN_actions, NN_termination):
        self.NN_options = NN_options
        self.NN_actions = NN_actions
        self.NN_termination = NN_termination
        self.options_weights = NN_options.get_weights()
        self.actions_weights = NN_actions.get_weights()
        self.termination_weights = NN_termination.get_weights()
        
    def save(self, lambdas, eta):
        self.NN_options.save('Triple_models/H_model_lambda_{}_eta_{}/NN_options'.format(lambdas,eta))
        self.NN_actions.save('Triple_models/H_model_lambda_{}_eta_{}/NN_actions'.format(lambdas,eta))
        self.NN_termination.save('Triple_models/H_model_lambda_{}_eta_{}/NN_termination'.format(lambdas,eta))
        
    def load(lambdas, eta):
        NN_options = keras.models.load_model('Triple_models/H_model_lambda_{}_eta_{}/NN_options'.format(lambdas,eta))
        NN_actions = keras.models.load_model('Triple_models/H_model_lambda_{}_eta_{}/NN_actions'.format(lambdas,eta))
        NN_termination = keras.models.load_model('Triple_models/H_model_lambda_{}_eta_{}/NN_termination'.format(lambdas,eta))
        
        return NN_options, NN_actions, NN_termination
        
    
class Experiment_design:
    def __init__(self, labels, TrainingSet, size_input, action_space, option_space, termination_space, N, zeta, mu, Triple_init, gain_lambdas,
                  gain_eta, env, max_epoch):
        self.labels = labels
        self.TrainingSet = TrainingSet
        self.size_input = size_input
        self.action_space = action_space
        self.option_space = option_space
        self.termination_space = termination_space
        self.N = N
        self.zeta = zeta
        self.mu = mu
        self.Triple_init = Triple_init
        self.gain_lambdas = gain_lambdas
        self.gain_eta = gain_eta
        self.env = env
        self.max_epoch = max_epoch
    


    
    