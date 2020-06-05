# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 03:54:31 2020

@author: Racehorse
"""


import numpy as np
import keras as K
import tensorflow as tf
from keras import optimizers, losses

import helper_functions


def get_optimizer(opt_string):
  if opt_string == 'adagrad':
    optimizer = optimizers.Adagrad
  elif opt_string == 'adam':
    optimizer = optimizers.Adam
  elif opt_string == 'adadelta':
    optimizer = optimizers.Adadelta
  elif type(opt_string)==str:
    raise ValueError('Optimizer not implemented!')
  else: optimizer = opt_string #If optimizer was passed directly
  return optimizer

def build_accelerated_model(DoDQN, input_dim, online_model, target_model, batch_size, 
               optimizer = 'adadelta', learning_rate = None, gamma = 1):
  
  optimizer = get_optimizer(optimizer)
  # action = K.Input(batch_shape = (1,), dtype = tf.int32, name = 'action')
  # reward = K.Input(batch_shape = (1,), dtype = tf.float32, name = 'reward')
  ard = K.Input(shape = (3,), dtype = tf.int32, name = 'ard')
  S_t = K.Input(shape = (input_dim,), name = 'S_t')
  S_t1 = K.Input(shape = (input_dim,), name = 'S_t1')
  Q_t = online_model(S_t)
  Q_t1 = online_model(S_t1)
  Q_tar = target_model(S_t1)
    
  if DoDQN:  #DoDQN Model
    class combine_Q(K.layers.Layer):
      def __init__(self, gamma,
                    **kwargs):
        super(combine_Q, self).__init__(**kwargs)
        self.supports_masking = False
        self.gamma = gamma
        
      def build(self, input_shape):
        input_dim = [shape[-1] for shape in input_shape]
        self.bias = None
        self.built = True
        super(combine_Q, self).build(input_shape)
                             
      def call(self, inputs):
        Q_t, ard, Q_t1, Q_tar  = inputs
        argmax = tf.expand_dims(tf.math.argmax(Q_t1, axis = 1), axis = -1)
        Q_t1_max = tf.gather(Q_tar, argmax, axis = 1, batch_dims = 1)
        new_Qta = tf.math.add(tf.math.multiply(self.gamma, Q_t1_max), ard[:,1])
        new_Q = tf.tensor_scatter_nd_update(Q_t, ard[:,0], new_Qta)
        return tf.stop_gradient(new_Q)
      


  else: #Q Learning
    class combine_Q(K.layers.Layer):
      def __init__(self, gamma,
                    **kwargs):
        super(combine_Q, self).__init__(**kwargs)
        self.supports_masking = False
        self.gamma = gamma
        
      def build(self, input_shape):
        self.index = tf.range(0, batch_size)
        self.bias = None
        self.built = True
        super(combine_Q, self).build(input_shape)
                            
      def call(self, inputs):
        Q_t, ard, Q_t1, Q_tar = inputs
        Q_t1_max = tf.math.reduce_max(Q_t1, axis = 1)
        new_Qta = tf.math.add(tf.math.multiply(self.gamma, Q_t1_max), tf.cast(ard[:,1], tf.float32))
        size = tf.shape(ard)[0]
        index = tf.range(0, size)
        indices= tf.stack([index, ard[:,0]], axis = 1)
        new_Q = tf.tensor_scatter_nd_update(Q_t, indices, new_Qta)
        return tf.stop_gradient(new_Q)

  class MAE(K.layers.Layer):
    def call(self, inputs):
      input1, input2 = inputs
      difference = tf.math.subtract(input1, input2)
      absolute = tf.math.abs(difference)
      return absolute
    
  new_Q = combine_Q(gamma)([Q_t, ard, Q_t1, Q_tar])
  # diff = K.layers.Subtract()([Q_t, new_Q])
  output = MAE()([Q_t, new_Q])
  # output = Q_t
  
  training_model = K.Model([S_t, ard, S_t1], [output])
  
  if learning_rate== None:
    training_model.compile(loss=losses.mean_absolute_error, optimizer = optimizer())
  else:
    training_model.compile(loss=losses.mean_absolute_error, optimizer = optimizer(
      lr = learning_rate))
  return training_model


def get_accelerated_update_strategy(action_space, training_model):
  def update_strategy(experience):
    """Update Q network"""
    helper_functions.timer['prep'].start()
    from_obs, ard, to_obs = zip(*experience)
    # Turn them into arrays (isntead of list of arrays)
    helper_functions.timer['prep_1'].start()
    from_obs_array = np.array(from_obs)
    to_obs_array = np.array(to_obs)
    helper_functions.timer['prep_1'].stop()
    helper_functions.timer['prep_2'].start()
    ard_array = np.array(ard)
    helper_functions.timer['prep_2'].stop()
    helper_functions.timer['prep_3'].start()
    inputs = [from_obs_array, ard_array, to_obs_array]
    helper_functions.timer['prep_3'].stop()
    helper_functions.timer['prep'].stop()
    helper_functions.timer['train'].start()
    loss = training_model.train_on_batch(inputs, 
                                          np.zeros((len(experience), action_space)))
    helper_functions.timer['train'].stop()
    return loss
  return update_strategy
    
def get_CPU_update_strategy(alpha, gamma, DoDQN, online_model, target_model):
  def update_strategy(experience):
    """Update Q network"""
    helper_functions.timer['prep'].start()
    from_obs, action, reward, to_obs, done = zip(*experience)
    # Turn them into arrays (isntead of list of arrays)
    helper_functions.timer['prep_1'].start()
    from_obs_array = np.array(from_obs)
    to_obs_array = np.array(to_obs)
    helper_functions.timer['prep_1'].stop()
    # Calculate Q Values
    helper_functions.timer['prep_2'].start()
    if DoDQN != 1:
      online_next_Q_values = np.array(online_model.predict_on_batch(to_obs_array))
    if DoDQN > 0:
      target_Q_values = np.array(target_model.predict_on_batch(to_obs_array))
    Q_values = np.array(online_model.predict_on_batch(from_obs_array))
    helper_functions.timer['prep_2'].stop()
    # Modify predictions based on rewards
    helper_functions.timer['prep_3'].start()
    if DoDQN==2:
      argmax = np.expand_dims(np.argmax(online_next_Q_values, axis = 1), axis = -1)
      max_for_next_obs = np.take_along_axis(target_Q_values, argmax, axis = 1)
    elif DoDQN == 1:
      max_for_next_obs = np.amax(target_Q_values, axis = 1)
    else:
      max_for_next_obs = np.amax(online_next_Q_values, axis = 1)
    for batch_index in range(len(experience)):
      Q_values[batch_index][action[batch_index]] *= 1-alpha
      if done[batch_index]:
        Q_values[batch_index][action[batch_index]] = reward[batch_index]
      else:
        calc_action_value = alpha*(reward[batch_index] + gamma*max_for_next_obs[batch_index])
        Q_values[batch_index][action[batch_index]]+=calc_action_value
    helper_functions.timer['prep_3'].stop()
    helper_functions.timer['prep'].stop()
    helper_functions.timer['train'].start()
    loss = online_model.train_on_batch(from_obs_array, 
                                            Q_values)
    helper_functions.timer['train'].stop()
    return loss
  return update_strategy

if __name__ == '__main__':
  from main import run_test
  run_test()