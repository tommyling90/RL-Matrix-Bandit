import yaml
import os
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
np.random.seed(config['defaults']['seed'])

folder = f"../{config[next(iter(config))]['save_folder']}"
if os.path.isdir(folder):
    raise "Folder already exists. Name another folder so to not overwrite existing graphs."
else:
    os.makedirs(folder, exist_ok=True)

rows = []
for key, value in config.items():
    if key == 'defaults':
        rounds = value['runs']
        horizon = value['horizon']
        player = value['player']
        continue

    matrix = np.array(value['matrix'])
    n_actions = len(matrix[0])
    matrices = generate_n_player_diag(player, n_actions, matrix) if is_diagonal(matrix) else generate_n_player(player, n_actions, matrix)

    results = run_one_game_experiments(value['name'], matrices, value['noise'][0], value['algos'], n_actions, rounds=rounds, horizon=horizon, n_agents=player)
    num_instances, num_players, num_rounds = results['all_plays'].shape
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
