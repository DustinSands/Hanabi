# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:07:12 2020

@author: Racehorse
"""


from Hanabi import hanabi_env as hb
import math
import matplotlib as plt

def wrapper(iterations, mode = 'standard', players = '3', suits = 'rgbyp'):
  """allows hanabi to run for iteration times with given settings
  stores and plots score over time.
  
  WIP / UNFINISHED"""
  num_epochs = 100
  epoch_length = math.floor(iterations / num_epochs)
  
  env = hb(players, suits, mode)
  for epoc in range(num_epochs):
    for iter in range(epoch_length):
      state, reward, done, info = env.reset()
      while done == False:
        #Run NN
        action = 'discard'
        specifics = 3
        state, reward, done, info = env.step(action, specifics)
      