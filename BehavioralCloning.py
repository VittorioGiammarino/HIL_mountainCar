#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:34:50 2020

@author: vittorio
"""
from tensorflow import keras
import numpy as np
import tensorflow as tf 
import numpy as np
import argparse
import os
import gym
import Simulation as sim


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

def AverageExpert(TrainingSet):
    trajs = 0
    for i in range(len(TrainingSet)):
        if TrainingSet[i,1]==0:
            trajs +=1
    average = len(TrainingSet)/trajs
    
    return average

def NN1(action_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input,)),
    keras.layers.Dense(action_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresBC/model_plotNN1.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model
    
def NN2(action_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input,)),
    keras.layers.Dense(action_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresBC/model_plotNN2.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
    return model

def NN3(action_space, size_input):
    model = keras.Sequential([
    keras.layers.Dense(300, activation='relu', input_shape=(size_input,)),
    keras.layers.Dense(action_space),
    keras.layers.Softmax()
    ])

    tf.keras.utils.plot_model(model, to_file='Figures/FiguresBC/model_plotNN3.png', 
                              show_shapes=True, 
                              show_layer_names=True,
                              expand_nested=True)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.Hinge(),
                  metrics=['accuracy'])
    
    return model

    
def EvaluationNN1(env, action_space, size_input, labels, TrainingSet, nSamples, max_epochs):
    
    average_NN = np.empty((0))
    success_percentageNN = np.empty((0))

    for i in range(len(nSamples)):
        model = NN1(action_space, size_input)
        model.fit(TrainingSet[0:nSamples[i],:], labels[0:nSamples[i]], epochs=50)
        T=100
        [trajNN,controlNN,flagNN]=sim.FlatPolicySim(env, model, max_epochs, T, size_input)
        length_trajNN = np.empty((0))
        for j in range(len(trajNN)):
            length_trajNN = np.append(length_trajNN, len(trajNN[j][:]))
        average_NN = np.append(average_NN,np.divide(np.sum(length_trajNN),len(length_trajNN)))
        success_percentageNN = np.append(success_percentageNN,np.divide(np.sum(flagNN),len(length_trajNN)))
        
    return average_NN, success_percentageNN   


def EvaluationNN2(env, action_space, size_input, labels, TrainingSet, nSamples, max_epochs):
    
    average_NN = np.empty((0))
    success_percentageNN = np.empty((0))

    for i in range(len(nSamples)):
        model = NN2(action_space, size_input)
        encoded = tf.keras.utils.to_categorical(labels)
        model.fit(TrainingSet[0:nSamples[i],:], encoded[0:nSamples[i],:], epochs=50)
        T=100
        [trajNN,controlNN,flagNN]=sim.FlatPolicySim(env, model, max_epochs, T, size_input)
        length_trajNN = np.empty((0))
        for j in range(len(trajNN)):
            length_trajNN = np.append(length_trajNN, len(trajNN[j][:]))
        average_NN = np.append(average_NN,np.divide(np.sum(length_trajNN),len(length_trajNN)))
        success_percentageNN = np.append(success_percentageNN,np.divide(np.sum(flagNN),len(length_trajNN)))
        
    return average_NN, success_percentageNN  


def EvaluationNN3(env, action_space, size_input, labels, TrainingSet, nSamples, max_epochs):
    
    average_NN = np.empty((0))
    success_percentageNN = np.empty((0))

    for i in range(len(nSamples)):
        model = NN3(action_space, size_input)
        encoded = tf.keras.utils.to_categorical(labels)
        model.fit(TrainingSet[0:nSamples[i],:], encoded[0:nSamples[i],:], epochs=50)
        T=100
        [trajNN,controlNN,flagNN]=sim.FlatPolicySim(env, model, max_epochs, T, size_input)
        length_trajNN = np.empty((0))
        for j in range(len(trajNN)):
            length_trajNN = np.append(length_trajNN, len(trajNN[j][:]))
        average_NN = np.append(average_NN,np.divide(np.sum(length_trajNN),len(length_trajNN)))
        success_percentageNN = np.append(success_percentageNN,np.divide(np.sum(flagNN),len(length_trajNN)))
        
    return average_NN, success_percentageNN  

