# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 01:49:30 2020

@author: Racehorse
"""
import numpy as np

from experience_replay import SequentialDequeMemory
import helper_functions


class agent():
  """Holds information for each player.  Retains previous observations, and returns an
  experience tuple for every round after first.  Has a final round function for
  storing it's previous obs, action, and (reward, and final obs) tuple returned by
  wrapper."""
  def __init__(self, env, model, policy, mem_size, action_map, playerID, integrated):
    self.memory = SequentialDequeMemory(size = mem_size)
    self.env = env
    self.model = model
    self.policy = policy
    self.action_map = action_map
    self.ID = playerID
    self.obs = None
    self.integrated = integrated
    
  def get_memory_batch(self, batch_size):
    """Get random experiences from memory for training."""
    return self.memory.get_random(batch_size)
  
  def add_to_memory(self, experience):
    self.memory.add_to_memory(experience)
  
  def play_turn(self, obs, reward):
    """Given a obs, the agent adds it to memory (it is the next_obs of the
    previous turn), chooses an action, stores the new obs, and passes the
  new obs back."""
    # If playing normally
    # assert self.ID == obs[0]
    helper_functions.timer['player'].start()
    if type(self.obs) != type(None):
      if self.integrated:
        experience = (self.obs, [self.policy_action, reward, 0], obs)
      else:
        experience = (self.obs, self.policy_action, reward, obs, 0)
      self.add_to_memory(experience)
    self.obs = obs
    processed = np.reshape(obs,(1,-1))
    helper_functions.timer['predict'].start()
    # action_values = self.model.predict(obs, steps = 1)[0]
    action_values = self.model.predict_on_batch(processed)[0]
    helper_functions.timer['predict'].stop()
    self.policy_action = self.policy(action_values, obs[0])
    action = self.action_map[self.policy_action]
    helper_functions.timer['env'].start()
    next_obs, reward, done, _ = self.env.step(*action)
    helper_functions.timer['env'].stop()
    next_obs = np.array(next_obs)
    helper_functions.timer['player'].stop()
    return next_obs, reward, done, self.policy_action
  
  def pass_final_obs(self, final_obs, reward):
    """Call this function when the game is done to pass the final obs (and 
    any associated rewards) to the player for storage."""
    if type(self.obs) != type(None):
      if self.integrated:
        experience = (self.obs, [self.policy_action, reward, 1], final_obs)
      else:
        experience = (self.obs, self.policy_action, reward, final_obs, 1)
      self.memory.add_to_memory(experience)
      self.obs = None
    
  def pure_exploit_turn(self, obs):
    processed = np.reshape(obs,(1,-1))
    action_values = self.model.predict_on_batch(processed)[0]
    exploit_action = np.argmax(action_values)
    action = self.action_map[exploit_action]
    next_obs, reward, done, _ = self.env.step(*action)
    next_obs = np.array(next_obs)
    return next_obs, reward, done, exploit_action