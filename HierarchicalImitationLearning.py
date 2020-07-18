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
import gym
from tensorflow import keras
import tensorflow.keras.backend as kb
import numpy as np
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


def NN_options(option_space):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(3,)),
    keras.layers.Dense(option_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='model_NN_options.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_actions(action_space):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(4,)),
    keras.layers.Dense(action_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='model_NN_actions.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    return model

def NN_termination(termination_space):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(4,)),
    keras.layers.Dense(termination_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='model_NN_termination.png', 
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
    
def TrainingSetTermination(TrainingSet,option_space):
    # Processing termination
    T = len(TrainingSet)
    TrainingSet_reshaped_termination = np.empty((int(option_space*(T-1)),4))
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
    
        
def TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels):
    TrainingSet_reshaped_actions = np.empty((int(option_space*(T)),4))
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

def TrainingSetPiLo(TrainingSet,o):
    # Processing termination
    T = len(TrainingSet)
    TrainingSet_PiLo = np.empty((T,4))
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
                    TrainingSet, eta, lambdas, gamma, T, option_space, labels):
    
    pi_b = NN_termination(TrainingSetTermination,training=True)
    pi_lo = NN_actions(TrainingSetActions,training=True)
    pi_hi = NN_options(TrainingSet,training=True)
    
    # Regularization 1
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
        
    # Regularization 2
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(option_space):
        ta = ta.write(i,kb.sum(-kb.sum(NN_actions(TrainingSetPiLo(TrainingSet,i))*kb.log(
                        NN_actions(TrainingSetPiLo(TrainingSet,i))),1)/T,0))
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
                               gamma, option_space, labels):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        
        with tf.GradientTape() as tape:
            weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
            tape.watch(weights)
            loss = RegularizedLossTot(gamma_tilde_reshaped, gamma_reshaped_options, gamma_actions_true, gamma_actions_false, 
                    NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                    TrainingSet, eta, lambdas, gamma, T, option_space, labels)

            
        grads = tape.gradient(loss,weights)
        optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
        optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
        optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
        print('options loss:', float(loss))
        
    return loss

def OptimizeLossAndRegularizerTotBatch(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
                               TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
                               TrainingSet, NN_options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                               gamma, option_space, labels, size_batch):
    
    n_batches = np.int(TrainingSet.shape[0]/size_batch)
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        for n in range(n_batches):
            with tf.GradientTape() as tape:
                weights = [NN_termination.trainable_weights, NN_actions.trainable_weights, NN_options.trainable_weights]
                tape.watch(weights)
                loss = RegularizedLossTot(gamma_tilde_reshaped[n*(T-1)*option_space:n*(T-1)*option_space+size_batch], 
                                          gamma_reshaped_options[n*(T-1):n*(T-1)+size_batch], gamma_actions_true, gamma_actions_false, 
                                          NN_termination, NN_options, NN_actions,TrainingSetTermination, TrainingSetActions, 
                                          TrainingSet, eta, lambdas, gamma, T, option_space, labels)

            
            grads = tape.gradient(loss,weights)
            optimizer.apply_gradients(zip(grads[0][:], NN_termination.trainable_weights))
            optimizer.apply_gradients(zip(grads[1][:], NN_actions.trainable_weights))
            optimizer.apply_gradients(zip(grads[2][:], NN_options.trainable_weights))
            print('options loss:', float(loss))
            
        
    return loss


def BaumWelch(labels, TrainingSet, action_space, option_space, termination_space, N, zeta, mu, lambdas, eta, Triple_weights_init):
    NN_Options = NN_options(option_space)
    NN_Actions = NN_actions(action_space)
    NN_Termination = NN_termination(termination_space)
    
    NN_Options.set_weights(Triple_weights_init.options_weights)
    NN_Actions.set_weights(Triple_weights_init.actions_weights)
    NN_Termination.set_weights(Triple_weights_init.termination_weights)
    
    T = TrainingSet.shape[0]
    TrainingSet_Termination = TrainingSetTermination(TrainingSet, option_space)
    TrainingSet_Actions, labels_reshaped = TrainingAndLabelsReshaped(option_space,T, TrainingSet, labels)

    for n in range(N):
        print('iter', n, '/', N)
    
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
        optimizer = keras.optimizers.Adamax(learning_rate=1e-3)
        epochs = 40 #number of iterations for the maximization step
        
        gamma_tilde_reshaped = GammaTildeReshape(gamma_tilde, option_space)
        gamma_actions_false, gamma_actions_true = GammaReshapeActions(T, option_space, action_space, gamma, labels_reshaped)
        gamma_reshaped_options = GammaReshapeOptions(T, option_space, gamma)
        loss = OptimizeLossAndRegularizerTot(epochs, TrainingSet_Termination, NN_Termination, gamma_tilde_reshaped, 
                                             TrainingSet_Actions, NN_Actions, gamma_actions_false, gamma_actions_true,
                                             TrainingSet, NN_Options, gamma_reshaped_options, eta, lambdas, T, optimizer, 
                                             gamma, option_space, labels)
    
        # loss = hil.OptimizeLoss(epochs, TrainingSetTermination, NN_termination, gamma_tilde_reshaped, 
        #                         TrainingSetActions, NN_actions, gamma_actions_false, gamma_actions_true,
        #                         TrainingSet, NN_options, gamma_reshaped_options, T, optimizer)

        print('Maximization done, Total Loss:',float(loss))#float(loss_options+loss_action+loss_termination))
        
    return NN_Termination, NN_Actions, NN_Options

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
    
  
def EvaluationBW(map, stateSpace, P, traj, control, ntraj, action_space, option_space, termination_space, 
                                                           N, zeta, mu, lambdas, eta, Triple_weights_init):
    averageBW = np.empty((0))
    success_percentageBW = np.empty((0))
    list_triple_weights = []

    for i in range(len(ntraj)):
        labels, TrainingSet = bc.ProcessData(traj[0:ntraj[i]][:],control[0:ntraj[i]][:],stateSpace)
        NN_Termination, NN_Actions, NN_Options = BaumWelch(labels, TrainingSet, 
                                                           action_space, option_space, termination_space, 
                                                           N, zeta, mu, lambdas, eta, Triple_weights_init)
        list_triple_weights.append(Triple_Weights(NN_Options.get_weights(), NN_Actions.get_weights(), NN_Termination.get_weights()))
        Trajs=100
        base=ss.BaseStateIndex(stateSpace,map)
        TERMINAL_STATE_INDEX = ss.TerminalStateIndex(stateSpace,map)
        [trajBW, controlBW, OptionBW, 
         TerminationBW, flagBW]=sim.HierarchicalStochasticSampleTrajMDP(P, stateSpace, NN_Options, NN_Actions, NN_Termination, 
                                                                        mu, 1000, Trajs, base, TERMINAL_STATE_INDEX, 
                                                                        zeta, option_space)
        length_trajBW = np.empty((0))
        for j in range(len(trajBW)):
            length_trajBW = np.append(length_trajBW, len(trajBW[j][:]))
        averageBW = np.append(averageBW,np.divide(np.sum(length_trajBW),len(length_trajBW)))
        success_percentageBW = np.append(success_percentageBW,np.divide(np.sum(flagBW),len(length_trajBW)))
      
    return averageBW, success_percentageBW, list_triple_weights
    
    
class Triple_Weights:
    def __init__(self, options_weights, actions_weights, termination_weights):
        self.options_weights = options_weights
        self.actions_weights = actions_weights
        self.termination_weights = termination_weights
    
    
    
    
    