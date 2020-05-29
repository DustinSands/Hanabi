# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:37:45 2020

@author: Racehorse
"""


import random
import time
import tensorflow as tf
from collections import deque
from tensorflow.keras.utils import Sequence

class Memory():
  def __init__(self, size = 2000):
    self.max_size = size
    
  def get_mem_size(self):
    """Returns number of experience tuples."""
    return len(self.memory)

class SequentialDequeMemory(Memory):
  def __init__(self, size=2000):
    self.max_size = size
    self.memory = deque(maxlen=size)
    
  def add_to_memory(self, experience):
    self.memory.append(experience)
    
  def get_random(self, batch_size):
    """Get batch for replay.
    """
    return random.sample(self.memory, batch_size)
  



class BatchMemorySequence(Sequence):
  """An unfinished class that was intended to do all of the processing
  ahead of time."""
  def __init__(self, size=2000):
    self.capacity = size  

  def add_to_memory(self, experience):
    tm = timer('memory add')
    self.memory.append(experience)
    tm.stop()
  
  def __len__(self):
    return int(np.ceil(len(self.x) / float(self.batch_size)))

  def __getitem__(self, idx, batch_size=64):
    batch_x = self.memory[idx * batch_size:(idx + 1) * batch_size][0]
    batch_y = self.memory[idx * batch_size:(idx + 1) * batch_size][1]

    return np.array([
      resize(imread(file_name), (200, 200))
        for file_name in batch_x]), np.array(batch_y)


if __name__ == '__main__':
  pass