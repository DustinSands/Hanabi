# Hanabi
The purpose of this code is to implement every version of reinforcement learning that I study so that I can 

- Learn RL
- Understand each implementation
- Compare their performance to deepen my understanding

This will be done with an environment (created by me) that plays Hanabi, but should be generalizable to any environment.

The current version only uses Q-learning.  One can create an instance by running
main.py and then the following in the console:

```
instance = wrapper()
```

and train with

```
instance.train(1)
```
## Features:

- Plots steps per episode and average reward on one graph every plot_frequency
- Model saving and loading
- Performance tracking (how long the wrapper is spending performing various tasks, both as a percentage and absolute value)


## Options currently implemented:

- Double DQN (versions 1 and 2 or disabled)
- Discrete agents (whether experiences are stored per player or as a single player)
- Customizable discount
- Customizable alpha (how much towards a new value each training batch should update)
- Customizable optimizer and learning rate
- "Greedy" and "Epsilon Greedy" Bandit algorithms
- Different numbers of players (2-5)
- Customizable hidden layers for model prediction
- Different batch size
- Customizable regularization (l1 only)
- Customizable memory size

See the comments for the options under wrapper for a description of available
options.

## In progress:
- GPU acceleration
- Dueling DQN
