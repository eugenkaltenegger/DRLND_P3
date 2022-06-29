# Udacity - Deep Reinforcement Learning - Project 1

## Project Description



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
