# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:22:40 2020

@author: Racehorse
"""

from wrapper import wrapper
import helper_functions


def run_test():
  ins = wrapper(batch_size = 512, 
                policy_type = 'epsilon_decay', 
                policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002},
                accelerated = True, Double_DQN_version = 0)
  ins.train(1)
  
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

  run_double_DQN_comparison()

  
  

