# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:56:49 2020

@author: Racehorse
"""
import time

timer={}

class time_tracker():
  """Tracks total time spent between start and stop.  
  """
  def __init__(self):
    self.on = False
    self.total = 0
  
  def start(self):
    assert self.on ==False
    self.on = True
    self.start_time = time.perf_counter()
  
  def stop(self):
    assert self.on == True
    self.on = False
    self.stop_time = time.perf_counter()
    elapsed = self.stop_time - self.start_time
    self.total += elapsed
    
  def print(self):
    print(f'{self.name} total elapsed:{self.total} s')
  
  def reset(self):
    self.on = False
    self.total = 0

def get_Q(playerID, batch_size, model):
  """
  Finds the average Q value across a batch for the given player.
  
  Created to be used for comparing how likely different players were to take 
  different actions.

  Parameters
  ----------
  playerID : int
    which player to take experiences from
  batch_size : int
  model : keras model
    The model used to make Q predictions given a state.

  Returns
  -------
  tempsum : array
    The average Q values

  """
  if instance ==None:
    instance = instance1
  obs, _, _, _, _ = zip(*instance.player[playerID].memory.get_random(batch_size))
  obs_tensors = tf.data.Dataset.from_tensors([*obs])
  Q_values = instance.online_model.predict([*obs_tensors], steps=1)
  tempsum = Q_values[0]
  for index in range(1,len(Q_values)):
    for action in range(len(Q_values[0])):
      tempsum[action] += Q_values[index][action]
  tempsum /= len(Q_values)
  return tempsum
  
def obs_to_readable(obs, players, suits = 'rgbyp'):
  """Given an observation (the state), prints in a human-readable format."""
  assert players ==3
  cards_per_hand = 5
  if players > 3:
    cards_per_hand = 4
  if players == 6:
    cards_per_hand = 3
  viewing_player = obs[0]
  deck_levels = obs[1:6]
  hints = obs[6]
  fuse = obs[7]
  print(f'Player:{viewing_player} | Deck levels:{deck_levels} | Hints:{hints} | Fuse:{fuse}')
  discards = obs[8:58]
  obs_pos = 58
  for tile in range(cards_per_hand):
    known = obs[obs_pos:obs_pos+11]
    obs_pos+=10
    clued = obs[obs_pos]
    obs_pos+=1
    suit_list, num_list = decode_known(known, suits)
    print(f'Tile{tile} suits:{suit_list}, numbers:{num_list}, Clued:{clued}')
  for player_offset in range(players-1):
    for tile in range(cards_per_hand):
      actual = obs[obs_pos:obs_pos+10]
      obs_pos+=10
      actual_suit, actual_num = decode_known(actual, suits)
      actual_str = (f'P+{player_offset+1} Tile{tile}: {actual_suit}{actual_num}')
      known = obs[obs_pos:obs_pos+10]
      obs_pos+=10
      clued = obs[obs_pos]
      obs_pos += 1
      suit_list, num_list= decode_known(known, suits)
      print(f'{actual_str} Known: suits:{suit_list}, numbers:{num_list}, Clued:{clued}')
      
  
def decode_known(bits, suits):
  """Converts the 11 bits of information to what is known about the hand in human-
readable format."""
  pos_suits =[suits[x] for x in range(len(suits)) if bits[x]]
  pos_num = [str(x +1) for x in range(5) if bits[x+5]]
  suitlist = ''.join(pos_suits)
  numlist = ''.join(pos_num)
  return suitlist, numlist

def print_random(instance):
  # Prints a random experience, showing the game state and what the agent did.
  # instance refers to the wrapper instance
  player = random.randrange(instance.players)
  obs, action, reward, next_obs, done = instance.player[
    player].get_memory_batch(1)[0]
  obs_to_readable(obs, instance.players)
  action = instance.action_map[action]
  print(f'Action:{action}, Reward:{reward}')
  obs_tensors = tf.data.Dataset.from_tensors([*(obs,)])
  Q_values = instance.online_model.predict([*obs_tensors], steps=1)[0]
  print(f'Clue Suit:{Q_values[0:5]}')
  print(f'Clue Num: {Q_values[5:10]}')
  print(f'Clue Suit:{Q_values[10:15]}')
  print(f'Clue Num: {Q_values[15:20]}')
  print(f'{Q_values[20:25]}')
  print(f'{Q_values[25:30]}')
  obs_to_readable(next_obs, instance.players)
  
  
Q = lambda: get_Q(0, 50, ins)

showme = lambda: print_random()

def print_times():
  """Helper function to print how long the agent has spent doing various tasks.
  Resets all timers at end. """
  percent = {}
  for name in timer_list:
    percent[name] = round(timer[name].total / timer['total'].total *100,1)
  print(f'Total time:{timer["total"].total}')
  print(f'Time spent playing rounds:{percent["step"]}%')
  print(f'-Player Decision Time:{percent["player"]}%')
  print(f'--Environment Time:{percent["env"]}%')
  print(f'--Prediction time:{percent["predict"]}%')
  print(f'Replay time:{percent["replay"]}%')
  print(f'-Recall Time:{percent["recall"]}%')
  print(f'-Prep Time:{percent["prep"]}%')
  print(f'--Making arrays Time:{percent["prep_1"]}%')
  print(f'--Q Value Time:{percent["prep_2"]}%')
  print(f'--Modifying targets Time Time:{percent["prep_3"]}%')
  print(f'-Train Time:{percent["train"]}%')
  print(f'Plot time:{percent["plot"]}%')
  for name in timer_list:
    timer[name].reset()

def print_integrated_times():
  """Helper function to print how long the agent has spent doing various tasks.
  Resets all timers at end. 
  
  Descriptions for the integrated model."""
  percent = {}
  for name in timer_list:
    percent[name] = round(timer[name].total / timer['total'].total *100,1)
  print(f'Total time:{timer["total"].total}')
  print(f'Time spent playing rounds:{percent["step"]}%')
  print(f'-Player Decision Time:{percent["player"]}%')
  print(f'--Environment Time:{percent["env"]}%')
  print(f'--Prediction time:{percent["predict"]}%')
  print(f'Replay time:{percent["replay"]}%')
  print(f'-Recall Time:{percent["recall"]}%')
  print(f'-Prep Time:{percent["prep"]}%')
  print(f'--Making arrays Time:{percent["prep_1"]}%')
  print(f'--ARD Time:{percent["prep_2"]}%')
  print(f'--Inputs Time:{percent["prep_3"]}%')
  print(f'-Train Time:{percent["train"]}%')
  print(f'Plot time:{percent["plot"]}%')
  for name in timer_list:
    timer[name].reset()
    
def reset_times():
  for name in timer_list:
    timer[name].reset()