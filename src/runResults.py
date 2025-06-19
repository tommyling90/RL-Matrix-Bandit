import yaml
import os
import pickle
import pandas as pd

from execute import Execute
from utils import *

def run_one_game_experiments(game_name, matrices, noise, algos, n_actions, rounds=500, horizon=1000, n_agents=2):
    results = {}
    title = f"{'×'.join(algos)}_{'_'.join(str(n) for n in noise)}_{game_name}"
    all_plays, all_regrets, all_rewards, all_explorations = Execute(
        rounds,
        horizon,
        n_agents,
        [None] * n_agents,
        game_name,
        n_actions
    ).getPDResult(
        matrices,
        algos,
        'normal',
        noise
    )
    results['title'] = title
    results['all_plays'] = all_plays
    results['all_regrets'] = all_regrets
    results['all_rewards'] = all_rewards
    results['all_explorations'] = all_explorations
    return results


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

for g in range(game_idx, len(games)):
    game = games[g]
    matrix = np.array(game['matrix'])
    n_actions = len(matrix[0])
    matrices = generate_n_player_diag(player, n_actions, matrix) if is_diagonal(matrix) else generate_n_player(
        player, n_actions, matrix)
    for r in range(run_idx if g == game_idx else 0, runs):
        for i in range(iter_idx if (g == run_idx and r == run_idx) else 0, horizon):
            results = run_one_game_experiments(game['name'], matrices, game['noise'][0], game['algos'], n_actions,
                                               rounds=r, horizon=i, n_agents=player)
            metrics.append({
                "run": r,
                "time_step": i,
                "results": results
            })

            if (i + 1) % save_every == 0 or i == horizon - 1:
                cp = {
                    "game_idx": g,
                    'run_idx': r,
                    'iter_idx': i + 1,
                    'metrics': metrics,
                }
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(cp, f)
                print(f"📝 Saved checkpoint: game={g}, run={r}, iter={i + 1}")

num_instances, num_players, num_rounds = metrics[-1]results['all_plays'].shape
title = results['title']

for k in range(num_instances):
    for i in range(num_players):
        row = {"title": title, "player": f"agent_{i+1}", "instance": f"instance_{k}", "n_actions": n_actions}
        for r in range(num_rounds):
            row[f"play_time{r}"] = results["all_plays"][k, i, r]
            row[f"reward_time{r}"] = results["all_rewards"][k, i, r]
            row[f"regret_time{r}"] = results["all_regrets"][k, i, r]
            row[f"exploration_time{r}"] = results["all_explorations"][k, i, r]
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(f"{folder}/output.csv", index=False)
with open(f"{folder}/config.yaml", 'w') as f:
    yaml.dump(config, f)
