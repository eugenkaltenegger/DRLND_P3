# Udacity - Deep Reinforcement Learning - Project 3

## Project Description

In this exercise the task is to create and train two agents to play tennis with each other.
The aim is to train the agents in such a way that the pass the ball to each other.
When an agent drops the ball it receives a negative reward of -0.01.
If an agent can pass the ball over the net the agent receives a positive reward of +0.1.
Therefore, the rewards are better the longer the agents play and pass the ball over the net to each other.

The environment vectors have the following dimensions for each agent:
- State vector size: 3 * 8 = 24 (3 frames of 8 observations stacked)
- Action vector size: 2

Therefore, the global view of the environment has a state vector of size 48 and an action vector of size 4.

After each episode the scores of each agent are summed up and the maximum of these two values is taken as episode score.
In order to solve the environment the score as described above over 100 episodes has the to be higher than 0.5 in average.

For this exercise it is a requirement that the agents utilize the [MADDPG](MADDPG_paper) architecture which is the multiagent version of [DDPG](DDPG_paper).

[DDPG_paper]: https://arxiv.org/pdf/1509.02971v6.pdf
[MADDPG_paper]: https://arxiv.org/pdf/1706.02275v4.pdf

## Solution Description

The environment is solved with the code and the agent in this repository.
The solution is build in various python files.
How to install the program and the dependencies is described in the `README.md` file in the section *Dependencies* and how operate the program is described in the `README.md` file in the section *Execution*.

The solution meets the requirement that the DDPG and the MADDPG architecture build the foundation of the agents.

### General Architecture

The general architecture of this project is according to requirements based on DDPG and MADDPG.
This approach requires an actor and an target_actor as well as a critic and a target_critic.
The networks which are not target networks (local network in the following) are utilized for training.
After each training step the new local network parameters are partially transferred to the target networks.
This ensures stability during the learning process.

#### Neuronal Network

The neuronal network is a group of layers of neurons and connections between the neurons of one layer to the neurons of the next layer.
The input layer is the entry point of information and the output layer is the exit point of the network where a decision is made.
Information that is passed into the network activates the neurons in the input layer which again activates the next layer and so on until the output layer is activated and a decision was made.
The intensity of the activation depends on the weights stored in the connection of the neurons.
During the training of the agent these weights are adopted to achieve the desired activation in the output layer corresponding to the input at the input layer.

### Specific Architecture

### Findings

### Improvements

The code provided in this repository solves the given environment.
Of course there is plenty of room for improvements:

- Setting manual seeds on every step that is based on random decisions would help easy the development process of any other further improvement.
- Instead of learning a step on an episode of the simulation it might be better to learn an episode of the simulation.
- The architecture provided in this repository has a weighted buffer.
  The weighting is based on the reward for single steps.
  Other weighting techniques might lead to better results.
- Further hyperparameter tuning might be beneficial.
  The unstable properties of multiple agents made the process of hyperparameter tuning difficult and time-consuming.
  Hyperparameter tuning with multiple runs on the same set of hyperparameters and averaging over the results might lead to a more stable set of hyperparameters and better performance.

Based on the high instability caused by multiple agents the number of episodes required to solve the environment has a high variety.

A beneficial 

## Summary

This exercise has been the most challenging exercise of the Udacity Deep Reinforcement Learning Nano Degree.
The high instability caused by multiple agents interacting makes the development process much more difficult.
Additionally, the high impact of noise (for example in the initial weight or on the drawing process when samples are taken from the buffer) makes it hard to determine the quality of the solution approach.

Nevertheless, I have been able to learn a lot during this course especially in the three main exercises.