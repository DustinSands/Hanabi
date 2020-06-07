# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 06:03:25 2020

@author: Racehorse
"""
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras as K
from keras import models, optimizers, losses

from behavior_policy import BehaviorPolicy
from envs import Hanabi as hb

import helper_functions, training_strategy
from model import create_Q_model
from agent import agent

# CONFIG
model_directory = r'C:\Users\Racehorse\Google Drive\Programming\ML\models'

helper_functions.timer_list = ['replay','player', 'predict', 'step',
              'plot', 'recall', 'train', 'prep', 'prep_1', 
              'prep_2', 'prep_3', 'total', 'env']

for name in helper_functions.timer_list:
  helper_functions.timer[name] = helper_functions.time_tracker()

def get_optimizer(opt_string, lr):
  if opt_string == 'adagrad':
    optimizer = optimizers.Adagrad
  elif opt_string == 'adam':
    optimizer = optimizers.Adam
  elif opt_string == 'adadelta':
    optimizer = optimizers.Adadelta
  elif opt_string == 'SGD':
    optimizer = optimizers.SGD
  elif type(opt_string)==str:
    raise ValueError('Optimizer not implemented!')
  else: optimizer = opt_string #If optimizer was passed directly
  if lr != None: optimizer = optimizer(learning_rate = lr)
  else: optimizer = optimizer()
  return optimizer

class wrapper():
  """Class for training the model. Parameters:
    name: name of agent for saving / loading from checkpoint (optional)
    discount: the discount factor (gamma, default 0.9)
    alpha: how much the Q values should be updated each training batch (default 1)
    lr: the learning rate (optimizer, default None of optimizer-specific default)
    behavior: the bandit algorithm to use (see behavior_policy to find which are implemented)
    policy_param: parameters for the behavior policy
    hidden_layers: list of int for hidden layers
    l1: l1 regularization parameter (0 is disabled)
    players: number of players
    suits: how many / what types of suits to use (environment variable, for variant games)
    optimizer: which optimizer to use
    mem_size: the maximum number of experiences to store per player
    max_steps: The max number of turns per game
    plot_frequency: number of epochs between plotting
    discrete_agents: True means that there is a different agent for each player
      (False means that the next state in experiences is from P+1, rather than
       the state P sees on their next turn)
    Double_DQN_version: Version of Double DQN to use (0 is disabled, 1 uses the 2010 paper, 2 uses 2015 paper)
    accelerated: Whether to make an accelerated training model (for GPU acceleration)
    games_per_epoch: How many episodes are in each epoch (affects model sync frequency, saving the model, calculating statistics, plotting)
    
    """
  def __init__(self, name = None, discount = 0.9, lr = None, alpha = 1,
          policy_type = 'greedy', policy_param = {'eps':0.05, 'min_eps':0.01,
          'eps_decay':0.9999}, env = hb, suits = 'rgbyp', players = 3, 
          mode = 'standard', hidden_layers = [200,200,200,150,100], batch_size = 512,
          l1 = 0, optimizer = 'adagrad', mem_size = 2000, max_steps = 130,
          plot_frequency = 1, discrete_agents = True, Double_DQN_version = 1, 
          accelerated = True, games_per_epoch = 100):
  
    self.name = name
    self.weights_dir = model_directory
    # if self.name == None:
    #   date = str(time.strftime('%m%d-%H%M'))
    #   self.name = f'{date}-{mode}-{suits}'
    if self.name != None:
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
    self.epoch_size = games_per_epoch
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
    self.Double_DQN_version = Double_DQN_version
    self.optimizer = get_optimizer(optimizer, lr)
    self.action_map = self._create_action_map()
    self.action_space = len(self.action_map)
    self.action_totals = [0]*self.action_space
    self.accelerated = accelerated
    move_func = self._create_valid_moves_function()
    self.policy = BehaviorPolicy(self.action_space, move_func, 
                                 policy_type = policy_type, 
                                 param = policy_param) 
    if self.name != None and os.path.exists(self.model_file):
      self.online_model = models.load_model(self.model_file)
      self.target_model = models.load_model(self.model_file)
      self.target_model.name = 'target_'+self.target_model.name
    else:
      self.online_model = create_Q_model(self.env, self.action_space, 
                                       self.optimizer, 
                                       self.hidden_layers,
                                       self.learning_rate, self.l1, 
                                       'online_model')
      self.online_model.name = 'online_model'
      self.target_model = create_Q_model(self.env, self.action_space, 
                                       self.optimizer, 
                                       self.hidden_layers,
                                       self.learning_rate, self.l1,
                                       'target_model')
      self.target_model.name = 'target_model'
    self._freeze_target_model()
    if self.accelerated:
      self.training_model = training_strategy.build_accelerated_model(self.Double_DQN_version, self.env.get_input_dim(), 
                                               self.online_model, self.target_model, self.batch_size*self.players, 
                                               self.optimizer, self.learning_rate, 
                                               self.gamma)
      self._update_online_model = training_strategy.get_accelerated_update_strategy(
        self.action_space, training_model = self.training_model, )
    else: self._update_online_model = training_strategy.get_CPU_update_strategy(
        alpha, self.gamma, Double_DQN_version, self.online_model, self.target_model)
    self.player = []
    for playerID in range(self.players):
      self.player.append(agent(self.env, self.online_model, self.policy.choose_action, 
                                     self.mem_size, self.action_map, playerID))

  def _freeze_target_model(self):
    for layer in self.target_model.layers:
      layer.trainable = False
      
  def train(self, epochs = 10, batch_size = None):
    """Trains the model for the specified number of epochs.  Optionally, can also
    be used to change the batch size."""
    helper_functions.timer['total'].start()
    if batch_size != None:
      self.batch_size = batch_size
    for epoch in range(epochs):
      #Reset stats
      epi_history = {}
      epi_history['steps'] = []
      epi_history['rewards'] = []
      epi_history['discounted_rewards'] = []
      epi_history['rps'] = []
      self.loss_history = []
      for episode in range(self.epoch_size):
        #Initialize and reset the episode
        reward = [0]*self.players
        episode_reward = 0
        discounted_episode_reward = 0
        current_obs, step_reward, done, info = self.env.reset()
        obs = np.array(current_obs)
        current_player = obs[0]
        helper_functions.timer['step'].start()
        for step in range(self.max_steps):
          # Passes the observation to the agent and has it play a turn.
          # The agent returns information from the environment
          obs, step_reward, done, action = self.player[
            current_player].play_turn(obs, reward[current_player])
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
          obs = np.array(obs)
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
      else: self.epoch_history['loss'].append(np.NaN)
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
      obs = np.array(current_obs)
      helper_functions.timer['step'].start()
      for step in range(self.max_steps):
        current_player = obs[0]
        # current_player = int(obs[0])
        obs, step_reward, done, action = self.player[
          current_player].pure_exploit_turn(obs)
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
    
    print(f'Rewards:{ave_rewards}, Steps:{ave_steps}, Loss:{self.loss_history[-1]}')
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
    loss = self._update_online_model(experience_batch)
    self.loss_history.append(loss)
    return True
  
  

  def save_model(self, name=None):
    """ Stores the model for future use."""
    if name is None:
      name = self.name
    model_file = os.path.join(self.weights_dir,name+'.h5')
    self.online_model.save(model_file, overwrite = True)
    
  def plot_training(self):
    #Plot steps and rewards per episode on two axes
    eps = round(self.policy.eps, 4)
    episodes = self.epoch_size*np.arange(len(self.epoch_history['steps']))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Steps in Episode', color = 'red')
    ax1.plot(episodes, self.epoch_history['steps'], color = 'red')
    ax1.set_ylim(top=80)
    # ax1.set_ylabel('Average Loss in Episode', color = 'red')
    # ax1.plot(episodes, self.epoch_history['loss'], color = 'red')
    # ax1.set_ylim(top=0.015)
    ax2 = ax1.twinx()
    ax2.set_ylim(bottom=-1, top=10)
    ax2.set_ylabel('Average Reward', color = 'blue')
    ax2.plot(episodes, self.epoch_history['rewards'], color='blue')
    fig.tight_layout()
    fig.dpi = 150
    plt.show()
    plt.close()