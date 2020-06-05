# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 01:47:34 2020

@author: Racehorse
"""
import random

from keras import models, optimizers, layers, regularizers, losses, initializers

def create_model(env, action_space, name = None, learning_rate = None, 
                       hidden_layers = [100,50,50,50,20,20], l1 = 0.0001,
                       optimizer = 'adagrad'):
  """Builds the Q model to be used for hanabi."""
  initializer = initializers.Ones()
  model = models.Sequential()
  model.add(layers.Dense(hidden_layers[0], 
                         input_dim = (env.get_input_dim()),
                         activation = 'relu',
                         kernel_regularizer=regularizers.l1(l1),
                         kernel_initializer = initializer, 
                         bias_initializer = initializer))
  if l1 == 0:
    for layer in hidden_layers[1:]:
      model.add(layers.Dense(layer, activation = 'relu',
                         kernel_initializer = initializer, 
                         bias_initializer = initializer))
    model.add(layers.Dense(action_space, activation = 'linear',
                         kernel_initializer = initializer, 
                         bias_initializer = initializer))
  else:
    for layer in hidden_layers[1:]:
      model.add(layers.Dense(layer, activation = 'relu',
                              kernel_regularizer=regularizers.l1(l1),
                         kernel_initializer = initializer, 
                         bias_initializer = initializer))
    model.add(layers.Dense(action_space, activation = 'linear', 
                           kernel_regularizer=regularizers.l1(l1),
                         kernel_initializer = initializer, 
                         bias_initializer = initializer))

  if learning_rate== None:
    model.compile(loss=losses.mean_absolute_error, optimizer = optimizer)
  else:
    model.compile(loss=losses.mean_absolute_error, optimizer = optimizer)
  return model