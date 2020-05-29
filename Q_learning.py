# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:22:40 2020

@author: Racehorse
"""

import copy
import math
import time
import os, pdb
import random
import timeit

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, losses, regularizers

from experience_replay import SequentialDequeMemory
from behavior_policy import BehaviorPolicyV2
from envs import Hanabi as hb

import helper_functions

def process_state(obs):
  return np.array(obs)

def build_hanabi_model(env, action_space, name = None, learning_rate = None, 
                       hidden_layers = [100,50,50,50,20,20], l1 = 0.0001, 
                       optimizer = 'adagrad'):
  """Builds a model to be used for hanabi."""
  model = models.Sequential()
  model.add(layers.Dense(hidden_layers[0], 
                         input_dim = (env.get_input_dim()),
                         activation = 'relu',
                         kernel_regularizer=regularizers.l1(l1)))
  if l1 == 0:
    for layer in hidden_layers[1:]:
      model.add(layers.Dense(layer, activation = 'relu'))
    model.add(layers.Dense(action_space, activation = 'linear'))
  else:
    for layer in hidden_layers[1:]:
      model.add(layers.Dense(layer, activation = 'relu',
                              kernel_regularizer=regularizers.l1(l1)))
    model.add(layers.Dense(action_space, activation = 'linear', 
                           kernel_regularizer=regularizers.l1(l1)))
  if optimizer == 'adagrad':
    optimizer = optimizers.Adagrad
  elif optimizer == 'adam':
    optimizer = optimizers.Adam
  elif optimizer == 'adadelta':
    optimizer = optimizers.Adadelta
  else: raise ValueError('optimizer not implemented!')
  if learning_rate== None:
    model.compile(loss=losses.mean_absolute_error, optimizer = optimizer())
  else:
    model.compile(loss=losses.mean_absolute_error, optimizer = optimizer(
      lr = learning_rate))
  return model

class player():
  """Holds information for each player.  Retains previous obs, and returns an
  experience tuple for every round after first.  Has a final round function for
  returning it's previous obs and action (final obs, reward to be added by 
  wrapper for experience tuple."""
  def __init__(self, env, model, policy, mem_size, action_map, playerID):
    self.memory = SequentialDequeMemory(size = mem_size)
    self.env = env
    self.model = model
    self.policy = policy
    self.action_map = action_map
    self.ID = playerID
    self.obs = None
    
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
    next_obs = process_state(next_obs)
    helper_functions.timer['player'].stop()
    return next_obs, reward, done, self.policy_action
  
  def pass_final_obs(self, final_obs, reward):
    """Call this function when the game is done to pass the final obs (and 
    any associated rewards) to the player for storage."""
    if type(self.obs) != type(None):
      experience = (self.obs, self.policy_action, reward, final_obs, 1)
      self.memory.add_to_memory(experience)
      self.obs = None
    
  def pure_exploit_turn(self, obs):
    processed = np.reshape(obs,(1,-1))
    action_values = self.model.predict_on_batch(processed)[0]
    exploit_action = np.argmax(action_values)
    action = self.action_map[exploit_action]
    next_obs, reward, done, _ = self.env.step(*action)
    next_obs = process_state(next_obs)
    return next_obs, reward, done, exploit_action
  
  
class wrapper():
  """Class for training the model. Requires parameters:
    name: name of agent for saving / loading from checkpoint
    iter: iterations
    discount: the discount factor
    lr: the learning rate (alpha)
    behavior: the behavior policy for exploring
    policy_param: parameters for the behavior policy
    hidden_layers: list of int for hidden layers
    l1: l1 regularization parameter (0 is disabled)
    """
  def __init__(self, name = None, discount = 0.9, lr = None, alpha = 1,
          policy_type = 'greedy', policy_param = {'eps':0.05, 'min_eps':0.01,
          'eps_decay':0.9999}, env = hb, suits = 'rgbyp', players = 3, 
          mode = 'standard', hidden_layers = [200,200,200,150,100], batch_size = 512,
          l1 = 0, optimizer = 'adagrad', mem_size = 2000, max_steps = 130,
          plot_frequency = 1, discrete_agents = True, DDQN = False):
  
    self.name = name
    # if self.name == None:
    #   date = str(time.strftime('%m%d-%H%M'))
    #   self.name = f'{date}-{mode}-{suits}'
    if self.name != None:
      self.weights_dir = r'C:\Users\Racehorse\Google Drive\Programming\ML\models'
      self.model_file = os.path.join(self.weights_dir,self.name+'.h5')
    self.env = hb.hanabi_env(players, suits, mode)
    self.iterations_done = 0
    self.gamma = discount
    self.learning_rate = lr
    self.alpha = alpha
    self.max_steps = max_steps
    self.policy_param = policy_param
    self.hidden_layers = hidden_layers
    self.discrete_agents = discrete_agents
    self.epoch = 0
    self.epoch_size = 100
    self.epoch_history = {}
    self.epoch_history['steps'] = []
    self.epoch_history['rewards'] = []
    self.epoch_history['discounted_rewards'] = []
    self.epoch_history['rps'] = []
    self.epoch_history['loss'] = []
    self.batch_size = batch_size
    self.plot_frequency = plot_frequency
    self.suits = suits
    self.mem_size = mem_size
    self.players = players
    self.mode = mode
    self.l1 = l1
    self.DDQN = DDQN
    self.optimizer = optimizer
    self.action_map = self._create_action_map()
    self.action_space = len(self.action_map)
    self.action_totals = [0]*self.action_space
    move_func = self._create_valid_moves_function()
    self.policy = BehaviorPolicyV2(self.action_space, move_func, 
                                 policy_type = policy_type, 
                                 param = policy_param) 
    if self.name != None and os.path.exists(self.model_file):
      self.online_model = models.load_model(self.model_file)
    else:
      self.online_model = build_hanabi_model(self.env, self.action_space, 
                                           self.name, self.learning_rate, 
                                           self.hidden_layers, self.l1, 
                                           self.optimizer)
    self.target_model = copy.copy(self.online_model)
    self.player = []
    
    for playerID in range(self.players):
      self.player.append(player(self.env, self.online_model, self.policy.choose_action, 
                                     self.mem_size, self.action_map, playerID))

  def _sync_models(self):
    """Sets target model weights to online model weights."""
    self.target_model.set_weights(self.online_model.get_weights())

  def _create_valid_moves_function(self):
    # Returns a function that gives a valid moves mask for the given player
    cards_per_hand = 5
    if self.players > 3:
      cards_per_hand = 4
    if self.players == 6:
      cards_per_hand = 3
      
    def movefunc(current_player):
      valid_moves = []
      for player_offset in range(self.players-1):
        player = (current_player+player_offset+1)%self.players
        valid = self.env.hand[player].get_valid_clues()
        valid = [*valid[0], *valid[1]]
        valid_moves.extend(valid)
      valid_moves.extend([1]*cards_per_hand*2)
      return valid_moves
    return movefunc
      
  
  def _create_action_map(self):
    """Specifies the action to be taken for action output by ID."""
    id = 0
    action_map = {}
    # Clues
    for player_offset in range(self.players-1):
      for suit in self.suits:
        action_map[id] = ('clue', (player_offset, suit))
        id += 1
      for num in range(1,6):
        action_map[id] = ('clue', (player_offset, num))
        id += 1
    # Plays, discards
    cards_per_hand = 5
    if self.players > 3:
      cards_per_hand = 4
    if self.players == 6:
      cards_per_hand = 3
    for slot in range(cards_per_hand):
      action_map[id] = ('play',slot)
      id += 1
      action_map[id] = ('discard',slot)
      id += 1
    return action_map
  
  def train(self, epochs = 10, batch_size = None):
    helper_functions.timer['total'].start()
    if batch_size != None:
      self.batch_size = batch_size
    for epoch in range(epochs):
      epi_history = {}
      epi_history['steps'] = []
      epi_history['rewards'] = []
      epi_history['discounted_rewards'] = []
      epi_history['rps'] = []
      self.loss_history = []
      for episode in range(self.epoch_size):
        reward = [0]*self.players
        episode_reward = 0
        discounted_episode_reward = 0
        current_obs, step_reward, done, info = self.env.reset()
        obs = process_state(current_obs)
        current_player = obs[0]
        helper_functions.timer['step'].start()
        for step in range(self.max_steps):
          
          obs, step_reward, done, action = self.player[
            current_player].play_turn(obs, reward[current_player])
          self.action_totals[action] += 1
          for playerID in range(self.players):
            reward[playerID] += step_reward
          reward[current_player] = step_reward
          episode_reward += step_reward
          discounted_episode_reward *= self.gamma
          discounted_episode_reward += step_reward
          if done:
            break
          if self.discrete_agents == True:
            current_player = obs[0]
        for playerID in range(self.players):
          obs= self.env.explicit_known(playerID)
          obs = process_state(obs)
          self.player[playerID].pass_final_obs(obs, reward[playerID])
        epi_history['steps'].append(step+self.players)
        epi_history['rewards'].append(episode_reward)
        epi_history['discounted_rewards'].append(
          discounted_episode_reward)
        epi_history['rps'].append(episode_reward/(step+self.players))
        helper_functions.timer['step'].stop()
        helper_functions.timer['replay'].start()
        self.replay_experience()
        helper_functions.timer['replay'].stop()
      if self.name != None:
        self.save_model()
      self._sync_models()
      #Plotting Section
      steps = sum(epi_history['steps'])/len(epi_history['steps'])
      rewards = round(sum(epi_history['rewards'])/len(epi_history['rewards']),2)
      rps = round(sum(epi_history['rps'])/len(epi_history['rps']), 4)
      discounted_rewards = round(sum(epi_history['discounted_rewards'])/len(
        epi_history['discounted_rewards']), 3)
      if self.loss_history != []:
        loss = round(sum(self.loss_history)/len(self.loss_history), 5)
        self.epoch_history['loss'].append(loss)
      self.epoch_history['steps'].append(steps)
      self.epoch_history['rewards'].append(rewards)
      self.epoch_history['discounted_rewards'].append(discounted_rewards)
      self.epoch_history['rps'].append(rps)
      
      if (epoch+1)%self.plot_frequency == 0:
        helper_functions.timer['plot'].start()
        self.plot_training()
        helper_functions.timer['plot'].stop()
    helper_functions.timer['total'].stop()
    # return self.epoch_history
  
  def evaluate(self, eval_size = 100):
    eval_history = {}
    eval_history['steps']=[]
    eval_history['rewards']=[]
    eval_history['discounted_rewards']=[]
    eval_history['rps']=[]
    for episode in range(eval_size):
      episode_reward = 0
      discounted_episode_reward = 0
      current_obs, step_reward, done, info = self.env.reset()
      obs = process_state(current_obs)
      helper_functions.timer['step'].start()
      for step in range(self.max_steps):
        current_player = obs[0]
        # current_player = int(obs[0])
        obs, step_reward, done, action = self.player[
          current_player].pure_exploit_turn(obs)
        self.action_totals[action] += 1
        episode_reward += step_reward
        discounted_episode_reward *= self.gamma
        discounted_episode_reward += step_reward
        if done:
          break
      eval_history['steps'].append(step+self.players)
      eval_history['rewards'].append(episode_reward)
      eval_history['discounted_rewards'].append(
        discounted_episode_reward)
      eval_history['rps'].append(episode_reward/(step+self.players))
      helper_functions.timer['step'].stop()
    ave_steps = sum(eval_history['steps'])/len(eval_history['steps'])
    ave_rewards = sum(eval_history['rewards'])/len(eval_history['steps']) 
    ave_discounted_rewards = sum(eval_history['discounted_rewards'])/len(eval_history['steps'])
    ave_rps = sum(eval_history['rps'])/len(eval_history['steps'])
    print(f'Rewards:{ave_rewards}, Steps:{ave_steps}')
    return ave_steps, ave_rewards, ave_discounted_rewards, ave_rps


  def replay_experience(self):
    """Retrieves experiences of batch size from each player, then trains the 
    model on the experiences."""
    if self.discrete_agents == False and self.player[0].memory.get_mem_size() >= self.batch_size*self.players:
      helper_functions.timer['recall'].start()
      experience_batch = self.player[0].get_memory_batch(
        self.batch_size*self.players)
      helper_functions.timer['recall'].stop()
      self._update_online_model(experience_batch)
      return True
    if self.player[-1].memory.get_mem_size() < self.batch_size:
      return False
    helper_functions.timer['recall'].start()
    experience_batch = []
    for playerID in range(self.players):
      experience_batch.extend(self.player[playerID].get_memory_batch(
        self.batch_size))
    helper_functions.timer['recall'].stop()
    self._update_online_model(experience_batch)
    return True
  
  def _update_online_model(self, experience):
    """Update Q network"""
    helper_functions.timer['prep'].start()
    from_obs, action, reward, to_obs, done = zip(*experience)
    # Turn them into arrays (isntead of list of arrays)
    helper_functions.timer['prep_1'].start()
    from_obs_array = np.array(from_obs)
    to_obs_array = np.array(to_obs)
    helper_functions.timer['prep_1'].stop()
    helper_functions.timer['prep'].stop()
    for step in range(1):
      helper_functions.timer['prep'].start()
      # Calculate Q Values
      helper_functions.timer['prep_2'].start()
      if self.DDQN:
        next_Q_values = np.array(self.online_model.predict_on_batch(to_obs_array))
      Q_values = np.array(self.online_model.predict_on_batch(from_obs_array))
      target_Q_values = np.array(self.target_model.predict_on_batch(to_obs_array))
      helper_functions.timer['prep_2'].stop()
      # Modify predictions based on rewards
      helper_functions.timer['prep_3'].start()
      if self.DDQN:
        argmax = np.expand_dims(np.argmax(next_Q_values, axis = 1), axis = -1)
        max_for_next_obs = np.take_along_axis(target_Q_values, argmax, axis = 1)
      else:
        max_for_next_obs = np.amax(target_Q_values, axis = 1)
      for index in range(len(experience)):
        Q_values[index][action[index]] *= 1-self.alpha
        if done[index]:
          Q_values[index][action[index]] += self.alpha*reward[index]
        else:
          calc_action_value = self.alpha*(reward[index] + self.gamma*max_for_next_obs[index])
          Q_values[index][action[index]]+=calc_action_value
      helper_functions.timer['prep_3'].stop()
      helper_functions.timer['prep'].stop()
      helper_functions.timer['train'].start()
      loss = self.online_model.train_on_batch(from_obs_array, 
                                              Q_values)
      helper_functions.timer['train'].stop()
    self.loss_history.append(loss)
    

  def save_model(self, name=None):
    """ Stores the model for future use."""
    if name is None:
      name = self.name
    model_file = os.path.join(self.weights_dir,name+'.h5')
    self.online_model.save(model_file, overwrite = True)
    
  def plot_training(self):
    #Plot steps and rewards on two axis
    eps = round(self.policy.eps, 4)
    episodes = self.epoch_size*np.arange(len(self.epoch_history['steps']))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Steps in Episode', color = 'red')
    ax1.plot(episodes, self.epoch_history['steps'], color = 'red')
    ax1.set_ylim(top=80)
    ax2 = ax1.twinx()
    ax2.set_ylim(bottom=-1, top=10)
    ax2.set_ylabel('Average Reward', color = 'blue')
    ax2.plot(episodes, self.epoch_history['rewards'], color='blue')
    fig.tight_layout()
    fig.dpi = 200
    plt.show()
    plt.close()


helper_functions.timer_list = ['replay','player', 'predict', 'step',
              'plot', 'recall', 'train', 'prep', 'prep_1', 
              'prep_2', 'prep_3', 'total', 'env']

for name in helper_functions.timer_list:
  helper_functions.timer[name] = helper_functions.time_tracker()


def print_ave_param():
  for condition in range(len(test_kwargs)):
    steps = sum(stats_data['steps'][condition])/len(stats_data['steps'][condition])  
    rewards = sum(stats_data['rewards'][condition])/len(stats_data['rewards'][condition]) 
    print(f'Condition {condition}: Steps {round(steps,2)}, Rewards {round(rewards,2)}, Disc {round(discounted_rewards,2)}')


if __name__=='__main__':
  instance = wrapper(discrete_agents = True, policy_type = 'epsilon_decay', policy_param ={'eps':0.1, 'eps_decay':0.9999, 'min_eps':0.002})
  instance.train(10)
  instance.evaluate(100)
  helper_functions.print_times()
  pass






  
    
      
    
      
    
    
    