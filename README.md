# Udacity - Deep Reinforcement Learning - Project 1

## Project Description

In this exercise the task is to greate and train two agents to play tennis with each other.
The aim is to train the agents in such a way that the pass the ball to each other.
When an agent drops the ball it receives a negative reward of -0.01.
If an agent can pass the ball over the net the agent receives a positive reward of +0.1.
Therefore, the rewards are better the longer the agents play.

The environment vectors have the following dimensions for each agent:
- State vector size: 3 * 8 = 24 (3 frames of 8 observations stacked)
- Action vector size: 2

Therefore, the global view of the environment has a state vector of size 48 and an action vector of size 4.

After each episode the scores of each agent are summed up and the maximum of these two values is taken as episode score.
In order to solve the environment the score as described above over 100 episodes has the to be higher than 0.5 in average.

For this exercise it is a requirement that the agents utilize the [MADDPG](MADDPG_paper) architecture which is the multiagent version of [DDPG](DDPG_paper).

[DDPG_paper]: https://arxiv.org/pdf/1509.02971v6.pdf
[MADDPG_paper]: https://arxiv.org/pdf/1706.02275v4.pdf

### Dependencies

**THIS SECTION ASSUMES THE READER/USER IS USING LINUX**

In order to operate this repository it is necessary to download the executables for the environment and to create a conda environment.
The training environment executables are free to download [here](download)

[download]: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip

The downloaded `.zip` files have to be extracted and placed in the folder `env`.
The resulting file structure should look like shown below:
```bash
├── env
│    ├── .gitkeep
│    └── Tennis_Linux
│        ├── Tennis_Data
│        ├── Tennis.x86
│        └── Tennis.x86_64
```

To create a conda environment and install the packages required for this repository run the following command:
```bash
conda env create --file requirements.yaml
```

This conda environment has to be activated with the following command:
```bash
conda activate kalteneger_p3_collaborative-competition
```

With the active conda environment and the installed dependencies the preparation to run the code is completed.

### Execution

**THIS SECTION ASSUMES THE READER/USER IS USING LINUX**

To execute the code run the following command in a terminal with the active conda environment:
```bash
python3 collaborative-competition.py <mode>
```

To code provided in this repository has three operation modes:
- `tune`: hyperparameter tuning, the list of hyperparameters set in form of lists in the file `hyperparameters_range.py` in the ordered dict is applied.
  The results of each hyperparameter combination are shown and finally the combination solving the environment after the least steps with the highest score is listed.
  The graph showing the scores of the best training with the best hyperparameter set is stored in `tuning.png`.
- `train`: training the agent, the agent is trained with the hyperparameters set in the file `hyperparameters.py` in the ordered dict.
  The graph for the score and the agent state are stored in `training.png` and `continuous-control.pth`.

To start the program the command could look like:
```bash
python3 collaborative-competition.py train
```
