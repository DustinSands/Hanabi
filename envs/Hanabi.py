import random
import numpy as np

# Initialize; game conditions

# Penalty for making an invalid move (like discarding with 8 clues or clueing
# with 0 clues)
invalid_move_penalty = -2
# Penalty for giving a bad clue (clueing 4 to a player with no 4s)
bad_clue_penalty = -1
# Percentage of score to subtract for ending game due to fuse
fuse_penalty = 10
# The number of each kind of tile
numbers = [1,1,1,2,2,3,3,4,4,5]

class _tile_:
  """Stores tile information"""
  def __init__(self, tile_gen):
    """Generates tile, intializes what is known"""
    self.suits = tile_gen.suits
    self.suit, self.number, self.id = next(tile_gen)
    self.possible_numbers = [1,1,1,1,1]
    self.possible_suits = [1 for suit in self.suits]
    self.clued = 0
    
  def clue(self, clue_value):
    """Modifies what is known based on clue"""
    if type(clue_value)==int:
      if clue_value==self.number:
        self.possible_numbers = [num==self.number-1 for num in range(5)]
        self.clued = 1
        return 1
      else: self.possible_numbers[clue_value-1] = False
    elif clue_value == self.suit:
      self.possible_suits=[clue_value==suit for suit in self.suits]
      self.clued = 1
      return 1
    else:
      self.possible_suits[self.suits.index(clue_value)]=0
    return 0
      
  def __str__(self):
    #Returns suit, number as string
    return f'{self.suit}{self.number}'
  
  def known(self):
    #Returns array of known by self info in format
    #[possible suits, possible numbers, clued]
    return [*self.possible_suits, *self.possible_numbers, self.clued]
  
  def seen(self):
    #Returns array of what other players see
    suit = [self.suit==suit for suit in self.suits]
    number = [num==self.number-1 for num in range(5)]
    return [*suit, *number]
  
    
    


class _hand:
  # Keeps track of tiles in hand, and knowledge of said tiles
  def __init__(self, tile_gen, hand_size):
    self.gen = tile_gen
    self.tile=[_tile_(tile_gen) for _ in range(hand_size)]
    self._find_valid_clues()
    self.size = hand_size
  
  def clue(self,clue_value):
    """Iterates over tiles in hand and passes on clue."""
    clues = 0
    for slot in range(len(self.tile)):
      clues += self.tile[slot].clue(clue_value)
    if clues == 0:
      return bad_clue_penalty
    return 0
      
  def _find_valid_clues(self):
    #Calculates valid clues to give
    #[suits, numbers]
    suits = [0,0,0,0,0]
    numbers = [0,0,0,0,0]
    for slot in range(len(self.tile)):
      suits[self.gen.suits.index(self.tile[slot].suit)]=1
      numbers[int(self.tile[slot].number)-1]=1
    self.valid_clues = [suits, numbers]
      
  def get_valid_clues(self):
    # Returns valid clues to give to this hand
    return self.valid_clues
  
  def discard_tile(self, tile_pos):
    # Discards tile and attempts to draw a new one.
    # Returns the discarded tile
    discard = self.tile.pop(tile_pos)
    try:
      self.tile= [_tile_(self.gen),*self.tile]
    except StopIteration:
      pass
    self._find_valid_clues()
    return discard
    
  def list(self):
    # List what tiles are in the hand
    return ([str(self.tile[x]) for x in range(len(self.tile))],
            [self.tile[x].clued for x in range(len(self.tile))])
  
  def viewable(self):
    # Returns full array of what is seen by external player, 
    # starting from slot 0
    # [suits, numbers, possible suits, possible numbers, clued]
    sparse = []
    for slot in range(len(self.tile)):
      sparse.extend(self.tile[slot].seen())    
      sparse.extend(self.tile[slot].known())
    for extra in range(self.size-len(self.tile)):
      extra += len(self.tile)
      sparse.extend([0]*21)
    return sparse
  
  def blinded(self):
    # Returns array of what the player knows about their own hand
    # [tile 0, tile 1, etc]
    # [possible suits, possible numbers, clued]
    info = []
    for slot in range(len(self.tile)):
      info.extend(self.tile[slot].known())
    for extra in range(self.size-len(self.tile)):
      extra += len(self.tile)
      info.extend([0]*11)
    return info

class _board_state():
  #Keeps track of play stack, clues, fuse, and discard
  def __init__(self, suits, deck_size, parent, mode='standard'):
    assert mode=='standard'
    self.board_state = [0 for suit in suits]
    self.suits = suits
    self.fuse = 0
    self.clock = 8
    self.discard = [0]*deck_size
    self.parent = parent
    
    
  def _increment_fuse(self):
    self.fuse += 1
    reward = -0.2
    if self.fuse>= 3:
      self.parent.done = True
      reward = -sum(self.board_state)*(fuse_penalty / 100)
    return reward
  
  def play(self, tile):
    # Takes a tile to play, and updates board state
    # Returns reward (or penalty)
    tile_suit = tile.suit
    tile_number = tile.number
    if self.board_state[self.suits.index(tile_suit)]==tile_number-1:
      self.board_state[self.suits.index(tile_suit)]=tile_number
      if tile_number ==5 & self.clock<8:
        self.clock +=1
      reward = 1
    else:
      reward = self._increment_fuse()
    return reward
      
  def total_score(self):
    # The current score
    return sum(self.board_state)
  
  def use_hint(self):
    self.clock -=1
    if self.clock < 0:
      self.clock = 0
      return invalid_move_penalty
    return 0
    
  def state(self):
    #Returns state of the board
    #Format: [play levels, clues, fuse, discard]
    state = [*self.board_state, self.clock, self.fuse, *self.discard]
    return state
  
  def discard_tile(self, card):
    # Store the discarded card in the discard pile
    # Also add a hint.  Returns penalty
    self.discard[card.id] = 1
    self.clock += 1
    if self.clock > 8:
      self.clock = 8
      return invalid_move_penalty
    return 0
  
# Tile Generator
class _tileset():
  """Generates tiles given suits possible."""
  def __init__(self, suits, calling_obj, mode='standard'):
    assert mode == 'standard', "only standard mode is implemented"
    self.parent = calling_obj
    self.pieces = []
    self.suits = suits
    id = 0
    for suit in suits:
      if suit == 'k':
        for number in range(5):
          self.pieces.append((suit, number, id))
          id += 1
      else: 
        for number in numbers:
          self.pieces.append((suit, number, id))
          id += 1
    self.total = len(self.pieces)
  
  # Gets the next tile.  Upon drawing last tile, sets lastround in parents
  def __next__(self):
    while len(self.pieces)>1:
      return self.pieces.pop(random.randrange(len(self.pieces)))
    if len(self.pieces)==1:
      self.parent._set_lastround()
      return self.pieces.pop(random.randrange(len(self.pieces)))
    else: raise StopIteration
    
  def remaining(self):
    return len(self.pieces)

class hanabi_env():
  """A Hanabi instance.  Create with settings for game mode, and can then be 
  reset for each new game (with the same settings).  Must be reset after
  initialization.
  
  Call step() for each player turn after that."""
  def __init__(self, players = 3,given_suits = 'bgrpy', mode='standard'):
    """Sets up initial parameters"""

    self.suits = given_suits
    self.players = players
    self.mode = mode
    self.hand_size = 5
    if players >= 4:
      self.hand_size = 4
    elif players == 6:
      self.hand_size = 3
  
  def reset(self):
    """Reset and initialize game state."""
    self.tileset = _tileset(self.suits,self)
    self.board = _board_state(self.suits, self.tileset.total, self, self.mode)
    self.hand=[_hand(self.tileset,self.hand_size) for _ in range(self.players)]
    self.current_player = 0
    self.last_player = -1
    self.done = 0
    info = None    
    return self.explicit_known(self.current_player), 0, 0, info
      
  def explicit_known(self,viewing_player):
    """Returns what the player sees.
    Format: [player#, play levels, clues, fuse, discard, Own hand, 
             next player hand, etc]
    hand format: [tile 1, tile 2, etc]
    [binary number possible, binary suit possible, clued]"""
    
    #Which player the perspective is from
    state = [viewing_player]
    #The board state, including discards
    state.extend(self.board.state())
    #What the player knows about their own hand
    state.extend(self.hand[viewing_player].blinded())
    #What the player sees in other hands
    for offset in range(self.players-1):
      offset += self.current_player
      offset %= self.players
      state.extend(self.hand[offset].viewable())
    return state
      
  def step(self, action, specifics):
    """Play, Discard, or Clue.  Returns observation, reward, done, and info.
    action is play, discard, or clue. info is:
    play: slot
    discard: slot
    clue: player_offset, clue (tuple int, str)"""
    
    # PLAY SECTION
    info = None
    reward = 0
    if self.last_player == self.current_player:
      #Last play!
      self.done = 1
    if action == 'play':
      # Remove card from hand, attempt to play, get reward
      tile = self.hand[self.current_player].discard_tile(specifics)
      reward = self.board.play(tile)
    elif action == 'discard':
      # Remove tile from hand and send to discard pile
      tile = self.hand[self.current_player].discard_tile(specifics)
      reward = self.board.discard_tile(tile)
    elif action == 'clue':
      #Find which player is clued, and pass on the clue
      reward = self.hand[(self.current_player+1+specifics[0])
                %self.players].clue(specifics[1])
      reward += self.board.use_hint()
    else: 
      raise ValueError(f'{action} is not a valid action!')
      
    # PREPARE FOR NEW PLAYER SECTION
    self.current_player +=1
    self.current_player %=self.players
    observations= self.explicit_known(self.current_player)
    #print(f'New player is {self.current_player}')
    return observations, reward, self.done, info
  
  def _set_lastround(self):
    """Initiate final round"""
    if self.last_player == -1:
      self.last_player = self.current_player
    
  def get_input_dim(self):
    """Finds the size of state and returns it"""
    try:
      self.state_size
    except: #if state size doesn't exist
      state, _, _, _ = self.reset()
      self.state_size = len(state)
    return self.state_size
    
  def print_seen(self):
    for offset in range(self.players-1):
      player = self.current_player+offset+1
      player %= self.players
      print(self.hand[player].list())

def run_test():
  debug = 1
  print('starting test')
  random.seed(0)
  #Class hanabi_env
  game = hanabi_env()
  game.reset()
  assert next(game.tileset) == ('r', 2, 23)
  assert game.hand[0].get_valid_clues() == [[1, 1, 1, 0, 1], [1, 1, 0, 1, 1]]
  assert game.hand[0].list() == (['r2', 'y5', 'r4', 'b1', 'g4'],[0,0,0,0,0])
  try:
    game.hand[0].clue(3)
    raise Exception('impossible clue gave no error!')
  except: pass
  game.hand[0].clue(2)
  assert game.hand[0].tile[0].known()==[1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]
  game.step('play',3)
  game.hand[0].list()
  assert game.hand[0].list()==(['g1', 'r2', 'y5', 'r4', 'g4'],[0,1,0,0,0])
  assert game.board.board_state==[1, 0, 0, 0, 0]
  game.step('clue',(1, 'g'))
  assert game.hand[0].list()==(['g1', 'r2', 'y5', 'r4', 'g4'], [1, 1, 0, 0, 1])
  game.step('clue',(1, 3))
  assert game.hand[1].list()==(['p3', 'p3', 'r5', 'r1', 'p4'], [1, 1, 0, 0, 0])
  assert game.board.clock==6
  game.step('discard',2)
  assert  game.hand[0].list()==(['b4', 'g1', 'r2', 'r4', 'g4'], [0, 1, 1, 0, 1])
  for i in range(30):
    game.step('discard',2)
  print('test finished successfully!')
  


if __name__ == '__main__':
  run_test()

            

            
