import yaml
import os
import pandas as pd

from execute import Execute
from utils import *
from pickleContext import PickleContext

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

defaults = config['defaults']
games = config['games']

np.random.seed(defaults['seed'])
runs = defaults['runs']
horizon = defaults['horizon']
player = defaults['player']
folder = f"../{defaults['save_folder']}"

if os.path.isdir(folder):
    raise "Folder already exists. Name another folder so to not overwrite existing graphs."
else:
    os.makedirs(folder, exist_ok=True)

checkpoint_file = f"{folder}/checkpoint.pkl"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "rb") as f:
        cp = pickle.load(f)
    game_idx = cp['game_idx']
    run_idx = cp['run_idx']
    iter_idx = cp['iter_idx']
    metrics = cp['metrics']
else:
    game_idx = run_idx = iter_idx = 0
    metrics = []

save_every = 5
ctx = PickleContext(game_idx, run_idx, iter_idx, save_every, metrics, checkpoint_file)

for g in range(game_idx, len(games)):
    game = games[f'game{g+1}']
    matrix = np.array(game['matrix'])
    n_actions = len(matrix[0])
    matrices = generate_n_player_diag(player, n_actions, matrix) if is_diagonal(matrix) else generate_n_player(
        player, n_actions, matrix)
    results = Execute(runs, horizon, player, [None] * player, game['name'], n_actions).get_one_game_result(
        matrices, game['algos'], ctx, g, 'normal', game['noise'][0])

cleaned_metrics = merge_metrics(ctx.metrics)
df = pd.DataFrame(cleaned_metrics)
df.to_csv(f"{folder}/output.csv", index=False)
with open(f"{folder}/config.yaml", 'w') as f:
    yaml.dump(config, f)
