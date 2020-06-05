# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:32:40 2020

@author: Racehorse
"""


import numpy as np

"""
Contains the bandit algorithm code
Note that when picking an exploration action, only valid moves are selected from
('greedy' and 'epsilon_decay' are implemented)

"""


class BehaviorPolicy:
  def __init__(self, action_space, valid_moves_function, 
               policy_type = 'greedy', param=
               {'eps':0.1}):
    self.current_policy = policy_type
    self.action_space = action_space
    self.policy_type = policy_type
    self.param = param
    if 'eps' not in param:
      raise ValueError('eps not specified')
    self.eps = self.param['eps']
    self.min_eps = None
    self.eps_decay = None
    self.move_func = valid_moves_function
    if self.policy_type == 'greedy':
      self.choose_action =  self.eps_greedy_pol
    elif self.policy_type == 'epsilon_decay':
      if ('min_eps' not in self.param) & ('eps_decay'not in self.param):
        raise ValueError('min eps or eps decay not in parameters')
      self.min_eps = self.param['min_eps']
      self.eps_decay = self.param['eps_decay']
      self.choose_action =  self.eps_decay_pol
    else: NotImplementedError('no policy matching given!')
     
    
  def eps_decay_pol(self, values, playerID):
    """Decays epsilon over time.
    Required parameters:
      min_eps: the minimum epsilon value
      eps_decay: the decay rate
      eps: the initial epsilon value
    """
    valid_moves = self.move_func(playerID)
    random_action_prob = self.eps / sum(valid_moves)
    action_vector = [x * random_action_prob for x in valid_moves]
    exploit_action_index = np.argmax(values)
    action_vector[exploit_action_index] += 1-self.eps
    chosen_action = np.random.choice(np.arange(self.action_space), p=
                                     action_vector)
    if self.eps > self.min_eps:
      self.eps *= self.eps_decay
    return chosen_action

  
  def eps_greedy_pol(self, values, playerID):
    """Constant epsilon value
    Required parameters:
      eps: the initial epsilon value
    """
    valid_moves = self.move_func(playerID)
    random_action_prob = self.eps / sum(valid_moves)
    action_vector = [x * random_action_prob for x in valid_moves]
    exploit_action_index = np.argmax(values)
    action_vector[exploit_action_index] += 1-self.eps
    chosen_action = np.random.choice(np.arange(self.action_space), p=
                                     action_vector)
    return chosen_action
