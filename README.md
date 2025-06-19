# Experiments for Deterministic vs. Stochastic Bandit Learning in Matrix Games

## Overview

This project implements several reinforcement learning algorithms, matrix games, and environments where the games are carried out. It also prints out the figures where the results of experiments are visualized.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repo-url>
cd project-directory

python3.10 -m venv PATH_TO_ENV/ENV_NAME
source PATH_TO_ENV/ENV_NAME/bin/activate
pip install -r ./requirements.txt
```

## Usage
```
python src/runAll.py
```
Note that with Apple M1 8-core 8Go-RAM it takes 25 minutes to run through the code and produce all the graphs used in the article.

## Directory Structure

```bash
├── src/                # Source code
├── Workshop/         
    ├── Figures/        # Figures of mean cumulative regrets and figures of joint selected actions
    ├── FiguresCompare/ # Figures of joint selected actions comparison
├── README.md           # This file
```


PG_WP
PG
PD
SG
CG

noise_levels = [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]]
algo_pairs = [
    ["UCB", "UCB"],
    ["KLUCB", "KLUCB"],
    ["TS", "TS"],
    ["UCB", "KLUCB"],
    ["UCB", "TS"]
]