# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 03:54:31 2020

@author: Racehorse
"""
import pdb

import numpy as np
import keras as K
import tensorflow as tf
from keras import optimizers, losses

import helper_functions




def build_accelerated_model(DoDQN, input_dim, online_model, target_model, batch_size, 
               optimizer = 'adagrad', learning_rate = None, gamma = 1):

  # action = K.Input(batch_shape = (1,), dtype = tf.int32, name = 'action')
  # reward = K.Input(batch_shape = (1,), dtype = tf.float32, name = 'reward')
  ard = K.Input(shape = (3,), dtype = tf.float32, name = 'ard')
  S_t = K.Input(shape = (input_dim,), name = 'S_t')
  S_t1 = K.Input(shape = (input_dim,), name = 'S_t1')
  Q_t = online_model(S_t)
  Q_t1 = online_model(S_t1)
  Q_tar = target_model(S_t1)

  
  class combine(K.layers.Layer):
    def __init__(self, gamma,
                    **kwargs):
      super(combine, self).__init__(**kwargs)
      self.supports_masking = False
      self.gamma = gamma
        
    def build(self, input_shape):
      self.bias = None
      self.built = True
      super(combine, self).build(input_shape)
        
  
  if DoDQN==2:  #DoDQN ver 2
    class combine_Q(combine):                             
      def call(self, inputs):
        Q_t, ard, Q_t1, Q_tar = inputs
        Q_t1_argmax = tf.math.argmax(Q_t1, axis = 1)
        reward = tf.cast(ard[:,1], tf.float32)
        action = tf.cast(ard[:, 0], tf.int32)
        done = tf.cast(ard[:, 2], tf.int32)
        Q_t1_max = tf.gather_nd(Q_tar, tf.expand_dims(Q_t1_argmax, axis = 1), batch_dims = 1)
        continue_candidates = tf.math.add(tf.math.multiply(self.gamma, Q_t1_max), reward)
        both_candidates = tf.stack([continue_candidates, reward], axis = 1)
        new_Qta = tf.gather_nd(both_candidates, tf.expand_dims(done, axis = 1), batch_dims = 1)
        size = tf.shape(ard)[0]
        index = tf.range(0, size)
        indices= tf.stack([index, action], axis = 1)
        new_Q = tf.tensor_scatter_nd_update(Q_t, indices, new_Qta)
        return tf.stop_gradient(new_Q)
      
  if DoDQN<=1:  #DoDQN ver 1 or Q learning
    class combine_Q(combine):
      def call(self, inputs):
        Q_t, ard, Q_tar = inputs
        Q_t1_max = tf.math.reduce_max(Q_tar, axis = 1)
        reward = tf.cast(ard[:,1], tf.float32)
        action = tf.cast(ard[:, 0], tf.int32)
        done = tf.cast(ard[:, 2], tf.int32)
        continue_candidates = tf.math.add(tf.math.multiply(self.gamma, Q_t1_max), reward)
        both_candidates = tf.stack([continue_candidates, reward], axis = 1)
        new_Qta = tf.gather_nd(both_candidates, tf.expand_dims(done, axis = 1), batch_dims = 1)
        size = tf.shape(ard)[0]
        index = tf.range(0, size)
        indices= tf.stack([index, action], axis = 1)
        new_Q = tf.tensor_scatter_nd_update(Q_t, indices, new_Qta)
        return tf.stop_gradient(new_Q)

  class MAE(K.layers.Layer):
    def call(self, inputs):
      input1, input2 = inputs
      difference = tf.math.subtract(input1, input2)
      absolute = tf.math.abs(difference)
      loss = tf.math.reduce_mean(absolute, axis = 1)
      return loss
    
  # Create target
  if DoDQN ==0:
    new_Q = combine_Q(gamma)([Q_t, ard, Q_t1])
  elif DoDQN ==1:
    new_Q = combine_Q(gamma)([Q_t, ard, Q_tar])
  else:
    new_Q = combine_Q(gamma)([Q_t, ard, Q_t1, Q_tar])
  # Calculate loss
  output = MAE(name = 'loss2')([Q_t, new_Q])
  # output = Q_t
  training_model = K.Model([S_t, ard, S_t1], [output])
  
  if learning_rate== None:
    training_model.compile(loss=my_loss_fn, optimizer = optimizer)
  else:
    training_model.compile(loss=my_loss_fn, optimizer = optimizer)
  return training_model

def my_loss_fn(y_true, y_pred):
  return y_pred
  # return tf.math.reduce_sum(y_pred, axis = -1)

def get_accelerated_update_strategy(action_space, training_model):
  def update_strategy(experience):
    """Update Q network"""
    helper_functions.timer['prep'].start()
    helper_functions.timer['prep_1'].start()
    from_obs, ard, to_obs = zip(*experience)
    # Turn them into arrays (isntead of list of arrays)
    helper_functions.timer['prep_1'].stop()
    helper_functions.timer['prep_2'].start()
    from_obs_array = np.array(from_obs)
    to_obs_array = np.array(to_obs)
    helper_functions.timer['prep_2'].stop()
    helper_functions.timer['prep_3'].start()
    ard_array = np.array(ard)
    inputs = [from_obs_array, ard_array, to_obs_array]
    helper_functions.timer['prep_3'].stop()
    helper_functions.timer['prep'].stop()
    helper_functions.timer['train'].start()
    loss = training_model.train_on_batch(inputs, 
                                         np.zeros(len(experience)))
    helper_functions.timer['train'].stop()
    return loss
  return update_strategy
    
def get_CPU_update_strategy(alpha, gamma, DoDQN, online_model, target_model):
  def update_strategy(experience):
    """Update Q network"""
    helper_functions.timer['prep'].start()
    from_obs, ard, to_obs = zip(*experience)
    action, reward, done = zip(*ard)
    # Turn them into arrays (instead of list of arrays)
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
  import main
  # main.run_test()