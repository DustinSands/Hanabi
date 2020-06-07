# Hanabi
A framework that allows for implementing various reinforcement learning models that I created so that I could:

- Learn RL
- Understand each implementation
- Compare their performance to deepen my understanding
- Deepen my understanding of tensorflow / keras

This will be done with an environment (created by me) that plays Hanabi, but should be generalizable to any environment.

Currently Q-learning is implemented and benefits from GPU acceleration.  I get roughly 25k samples trained / second on my desktop, and 10k / sec on my laptop.  

One can create an instance by running main.py and then the following in the console:

```
instance = wrapper()
```

and train with

```
instance.train(10)
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
- Full GPU acceleration (including target generation)
- Dueling DQN
