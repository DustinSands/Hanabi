# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 09:00:33 2020

@author: Racehorse
"""
import random

import numpy as np
import tensorflow as tf

import helper_functions
from wrapper import wrapper

def compare_accelerated(seed, DoDQN, done, epochs):
  """Compare the performance of the accelerated and normal models.
  Trains both models for number of epochs, picks a experience tuple, and prints
  Q values as well as loss function for debug purposes.  Both models should give
  same results."""
  global experience
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  ins = wrapper(batch_size = 1, 
                policy_type = 'epsilon_decay', 
                policy_param ={'eps':0, 'eps_decay':0.9999, 'min_eps':0.002},
                accelerated = True, Double_DQN_version = DoDQN, optimizer = 'adadelta',
                # games_per_epoch = 1, 
                hidden_layers =[100,100,100,100,100])
  ins.train(epochs)
  if epochs > 0:
    experience = ins.player[0].get_memory_batch(1)
    from_obs, ard, to_obs = zip(*experience)
    action = ard[0][0]
    reward = ard[0][1]
    done = ard[0][2]
    state = np.array(from_obs)
    state2 = np.array(to_obs)
    print(f'Done {done}')
  else:
    from_obs, _, _, _ = ins.env.reset()
    to_obs, _, _, _ = ins.env.reset()
    state = np.array([from_obs])
    state2 = np.array([to_obs])
    action = 0
    reward = 2
  print(ins.online_model.predict(state))
  print(ins.online_model.predict(state2))
    # print([state, np.array([[action,reward,done]]), state2], [0])
  loss = ins.training_model.evaluate([state, np.array([[action,reward,done]]), state2], [done])
  print(loss)
  
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  ins = wrapper(batch_size = 1, 
                policy_type = 'epsilon_decay', 
                policy_param ={'eps':0, 'eps_decay':0.9999, 'min_eps':0.002},
                accelerated = False, Double_DQN_version = DoDQN, optimizer = 'adadelta',
                # games_per_epoch = 1, 
                hidden_layers =[100,100,100,100,100])
  ins.train(epochs)
  print(ins.online_model.predict(state))
  print(ins.online_model.predict(state2))
  from_obs_array = state
  to_obs_array = state2
  alpha = 1
  gamma = 0.9
  if DoDQN != 1:
    online_next_Q_values = np.array(ins.online_model.predict_on_batch(to_obs_array))
  if DoDQN > 0:
    target_Q_values = np.array(ins.target_model.predict_on_batch(to_obs_array))
  Q_values = np.array(ins.online_model.predict_on_batch(from_obs_array))
  if DoDQN==2:
    argmax = np.expand_dims(np.argmax(online_next_Q_values, axis = 1), axis = -1)
    max_for_next_obs = np.take_along_axis(target_Q_values, argmax, axis = 1)
  elif DoDQN == 1:
    max_for_next_obs = np.amax(target_Q_values, axis = 1)
  else:
    max_for_next_obs = np.amax(online_next_Q_values, axis = 1)

  Q_values[0][action] *= 1-alpha
  if done:
    Q_values[0][action] = reward
  else:
    calc_action_value = alpha*(reward + gamma*max_for_next_obs[0])
    Q_values[0][action]+=calc_action_value
  # print(from_obs_array, Q_values)
  loss = ins.online_model.evaluate(from_obs_array, Q_values)
  print(loss)