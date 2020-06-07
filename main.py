# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:22:40 2020

@author: Racehorse
"""
import random

import numpy as np
import tensorflow as tf

from wrapper import wrapper
import helper_functions
import tests


def run_test():
  random.seed(0)
  np.random.seed(0)
  tf.random.set_seed(0)
  ins = wrapper(batch_size = 512, mem_size = 5000,
                policy_type = 'epsilon_decay', 
                policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002},
                accelerated = True, Double_DQN_version = 0, optimizer = 'adadelta')
  ins.train(5)
  helper_functions.reset_times()
  ins.train(10)
  helper_functions.print_integrated_times()
  
def run_double_DQN_comparison():
  # Repeatable testing of standard training model to compare Double DQN
  for version in range(3):
    step_array = []
    reward_array = []
    for i in range(3):
      ins = wrapper(batch_size = 512, 
                    policy_type = 'epsilon_decay', 
                    policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002},
                    accelerated = True, Double_DQN_version = version,
                    plot_frequency = 10)
      ins.train(500)
      steps, rewards, _, _ = ins.evaluate(100)
      step_array.append(steps)
      reward_array.append(rewards)
    print(f'Version {version} Rewards: {sum(reward_array)/len(reward_array)} Steps:{sum(step_array)/len(step_array)}')
    helper_functions.print_times()

def run_accelerated_comparison():
  # Repeatable testing of standard training model to compare Double DQN
  step_array = []
  reward_array = []
  random.seed(0)
  np.random.seed(0)
  tf.random.set_seed(0)
  for i in range(1):
    ins = wrapper(batch_size = 512, 
                  policy_type = 'epsilon_decay', 
                  policy_param ={'eps':0.0, 'eps_decay':0.9999, 'min_eps':0.002},
                  accelerated = True, Double_DQN_version = 1)
    ins.train(10)
    steps, rewards, _, _ = ins.evaluate(100)
    step_array.append(steps)
    reward_array.append(rewards)
  print(f'Accelerated Rewards: {sum(reward_array)/len(reward_array)} Steps:{sum(step_array)/len(step_array)}')
  helper_functions.print_integrated_times()
  step_array = []
  reward_array = []
  random.seed(0)
  np.random.seed(0)
  tf.random.set_seed(0)
  for i in range(1):
    ins = wrapper(batch_size = 512, 
                  policy_type = 'epsilon_decay', 
                  policy_param ={'eps':0.0, 'eps_decay':0.9999, 'min_eps':0.002},
                  accelerated = False, Double_DQN_version = 1)
    ins.train(10)
    steps, rewards, _, _ = ins.evaluate(100)
    step_array.append(steps)
    reward_array.append(rewards)
  print(f'Standard Rewards: {sum(reward_array)/len(reward_array)} Steps:{sum(step_array)/len(step_array)}')
  helper_functions.print_times()
    
if __name__=='__main__':
  # tests.compare_accelerated(0, 2, 20, done = 0)
  run_double_DQN_comparison()
  # run_accelerated_comparison()
  # run_test()
  # helper_functions.test_accelerated()
  pass
  
  

