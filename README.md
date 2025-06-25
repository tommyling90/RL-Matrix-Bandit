# Reinforcement Learning Framework for Experiments with Agent Learning in Matrix Games

## Overview

This project implements several reinforcement learning algorithms, matrix games, and environments where the games are carried out.
Some important features include game configuration by user, extending the games to multiple players, checkpointing, and etc.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repo-url>
cd project-directory

python3.10 -m venv PATH_TO_ENV/ENV_NAME
source PATH_TO_ENV/ENV_NAME/bin/activate
pip install -r ./requirements.txt
```

## Documentation

### 💡General Idea
In this framework, the generation of results, statistics, and figures is separated and modularized.
The user can run `runResults.py` to first generate results, then run `runStats.py` to generate the stats, and then run `runFigures.py` to generate figures.
Notice that when executing `runResults.py`, the user can interrupt the experiment at any moment, and the results to date would be saved in `pkl` files.
When he resumes, the experiments will pick up where he left off. This is called *checkpointing*.

The `pkl` files are saved to the directory `project_root/{folder}/pkl`.
Notice the `folder` in this path is provided by user in `config.yaml` - see Configuration section below.
These `pkl` files are used in order to generate the csv file - see Saving CSV below.

Along with the `pkl` files, `config.yaml` and `output.csv` will also be saved to the folder.
`config.yaml` is saved in order to provide the user an idea of what configurations he was running in that specific experiment.

### ✅ Core Concepts
#### Agents and Metrics
Each experiment involves multiple agents interacting over a number of iterations.
During each iteration, the framework records the following metrics:

- play (action chosen by the agent)
- reward (reward for the agent at one given time step)
- regret_time (regret of the agent at one given time step)
- exploration_time (whether the agent explored at one given time step. This is boolean represented with 0 and 1)

These metrics are stored per agent and per iteration.

### ⚙️ Configuration

Key parameters are loaded from a config.yaml file that the user needs to provide.
Refer to the existing `config.yaml` file to see what and how the parameters should be provided.
It is STRONGLY recommended to follow the same structure.

Notice the game names available are:
- PG_WP
- PG
- PD
- SG
- CG

The `noise` parameter should always follow this pattern: `[0.0, {noise_level_tested}]`

Algorithms available are:
- UCB
- KLUCB
- TS

In `defaults`, the `player` parameter specifies the number of players in the game. Note that this number MUST match the length of the `algos` param in game.

### 📦 Checkpointing
#### Save Strategy
- Checkpoints are saved every `save_every` iterations. The `save_every` variable can be found in `runResults.py` and can be modified as the user sees fit.
- At each save point, the following are recorded:
  - `game_idx`, `run_idx`, `iter_idx`
  - A list of flattened metric dicts called `delta`. Note that in `delta`, only the metrics of that certain checkpoint are recorded, in order to speed up the execution.

Each checkpoint is saved as a separate `.pkl` file named: `cp_game{g}_run{r}_iter{i+1}.pkl`

#### 📊 Saving CSV

Metrics of each checkpoint are stored in each `.pkl` file and then aggregated using the function `aggregate_metrics_from_pkl`.
This function can be found in `runResults.py`. Note that the default is to run this function to generate the csv once the experiments are done.

However, if the user has interrupted the experiments and wish to aggregate the data to date, he can still use this function to generate a csv that contains the data only to date.

## Directory Structure

```bash
├── src/                # Source code
├── Figures/         
    ├── Test/           # Folder specified in yaml to accommodate the relevant data for that experiment
        ├── pkl/        # Folder that contains all pkl files
        ├── config.yaml # A copy of config.yaml as reference to the configurations experimented with
        ├── output.csv  # Results of the experiment
├── config.yaml         # Original config.yaml that the user should provide
├── README.md           # This file
```