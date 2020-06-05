# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:22:40 2020

@author: Racehorse
"""
import random

import numpy as np

from wrapper import wrapper
import helper_functions


def run_test():
  ins = wrapper(batch_size = 512, 
                policy_type = 'epsilon_decay', 
                policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002},
                accelerated = True, Double_DQN_version = 0, optimizer = 'adadelta')
  ins.train(1)
  
def compare_accelerated(DoDQN, done, epochs):
  random.seed(0)
  reward = 1
  ins = wrapper(batch_size = 1, 
                policy_type = 'epsilon_decay', 
                policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002},
                accelerated = True, Double_DQN_version = DoDQN, optimizer = 'SGD',
                games_per_epoch = 1)
  # ins.train(epochs)
  ins.train(1)
  ins.train(1)
  state, _, _, _ = ins.env.reset()
  state2, _, _, _ = ins.env.reset()

  state = np.array([state])
  state2 = np.array([state2])
  print(ins.online_model.predict(state))
  loss = ins.training_model.evaluate([state, np.array([[0,reward,done]]), state2], [0])
  print(loss)
  ins = wrapper(batch_size = 1, 
                policy_type = 'epsilon_decay', 
                policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002},
                accelerated = False, Double_DQN_version = DoDQN, optimizer = 'SGD',
                games_per_epoch = 1)
  # ins.train(epochs)
  ins.train(1)
  ins.train(1)
  print(ins.online_model.predict(state))
  from_obs_array = state
  to_obs_array = state2
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

  Q_values[0][0] *= 1-1
  if done:
    Q_values[0][0] = reward
  else:
    calc_action_value = 1*(reward + 0.9*max_for_next_obs[0])
    Q_values[0][0]+=calc_action_value
  loss = ins.online_model.evaluate(from_obs_array, Q_values)
  print(loss)

  
def run_double_DQN_comparison():
  # Repeatable testing of standard training model to compare Double DQN
  for version in range(3):
    step_array = []
    reward_array = []
    for i in range(10):
      ins = wrapper(batch_size = 512, 
                    policy_type = 'epsilon_decay', 
                    policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002},
                    accelerated = False, Double_DQN_version = version)
      ins.train(100)
      steps, rewards, _, _ = ins.evaluate(100)
      step_array.append(steps)
      reward_array.append(rewards)
    print(f'Version {version} Rewards: {sum(reward_array)/len(reward_array)} Steps:{sum(step_array)/len(step_array)}')
    helper_functions.print_times()

if __name__=='__main__':
  compare_accelerated(0, 0, 2)
  # run_double_DQN_comparison()
  pass
  
  

