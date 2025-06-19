import numpy as np
import pickle
from collections import defaultdict

def flatten_metrics(title, player, instance, n_actions, metrics_dict):
    row = {
        "title": title,
        "player": player,
        "instance": instance,
        "n_actions": n_actions
    }

    for key, val in metrics_dict.items():
        for t, value in enumerate(val):
            row[f"{key}{t}"] = value
    return row

def merge_metrics(ctx_metrics):
    merged = defaultdict(lambda: {
        "title": None,
        "player": None,
        "instance": None,
        "n_actions": None,
        "metrics": defaultdict(list)
    })

    for row in ctx_metrics:
        key = (row["title"], row["player"], row["instance"], row["n_actions"])
        entry = merged[key]
        entry["title"], entry["player"], entry["instance"], entry["n_actions"] = key

        for k, v in row.items():
            if k in {"title", "player", "instance", "n_actions"}:
                continue
            entry["metrics"][k].append(v)

    # Now flatten the metrics dict into separate keys like play_time0, play_time1, ...
    flat_rows = []
    for entry in merged.values():
        flat_row = {
            "title": entry["title"],
            "player": entry["player"],
            "instance": entry["instance"],
            "n_actions": entry["n_actions"]
        }

        for metric_name, values in entry["metrics"].items():
            for i, v in enumerate(values):
                flat_row[f"{metric_name}{i}"] = v

        flat_rows.append(flat_row)

    return flat_rows

def save_pickle(ctx, horizon, g, r, i, plays, exploration_list, regrets, rewards, title, n_actions):
    if (i + 1) % ctx.save_every == 0 or i == horizon - 1:
        start = i - (i % ctx.save_every)
        end = i + 1
        for agent_id in range(plays.shape[0]):
            ctx.metrics.append(flatten_metrics(
                title=title,
                player=f"agent_{agent_id}",
                instance=f"instance_{r}",
                n_actions=n_actions,
                metrics_dict={
                    "play_time": plays[agent_id, start:end].tolist(),
                    # "reward_time": rewards[agent_id, i],
                    # "regret_time": regrets[agent_id, i],
                    "exploration_time": exploration_list[agent_id, start:end].tolist(),
                }
            ))

        cp = {
            "game_idx": g,
            'run_idx': r,
            'iter_idx': i + 1,
            'metrics': ctx.metrics,
        }
        with open(ctx.cp_file, "wb") as f:
            pickle.dump(cp, f)
        print(f"📝 Saved checkpoint: game={g+1}, run={r}, iter={i + 1}")


def generate_n_player_PD(n, reward_matrix):
    # 2 pcq trahir vs trahir pas
    shape = (2,) * n
    payoffs = [np.zeros(shape) for _ in range(n)]

    for actions in np.ndindex(shape):
        betray_count = sum(actions)

        for i in range(n):
            choice_i = actions[i]
            others = list(actions[:i] + actions[i + 1:])
            others_betray = sum(others)

            if betray_count == 0:
                val = reward_matrix[0,0]  # all cooperate
            elif betray_count == n:
                val = reward_matrix[1,1]  # all betray
            elif choice_i == 1 and others_betray == 0:
                val = reward_matrix[1,0]  # lone betrayer
            elif choice_i == 0 and others_betray == n - 1:
                val = reward_matrix[0,1]  # lone cooperator
            elif choice_i == 1:
                val = (1+reward_matrix[0,0])/2  # partial betrayal: betrayer reward
            else:
                val = (reward_matrix[1,1])/2  # partial betrayal: cooperator punished

            payoffs[i][actions] = val

    return payoffs

def generate_n_player_diag(n_players, k_actions, reward_matrix):
    shape = (k_actions,) * n_players
    reward_tensor = np.zeros(shape)

    for action_combo in np.ndindex(shape):
        if all(a == action_combo[0] for a in action_combo):
            reward_tensor[action_combo] = reward_matrix[action_combo[0], action_combo[0]]
        else:
            reward_tensor[action_combo] = 0.0

    return [reward_tensor] * n_players

def generate_n_player(n_players, k_actions, reward_matrix):
    shape = (k_actions,) * n_players
    reward_tensor = np.zeros(shape)

    for action_combo in np.ndindex(shape):
        if all(a == action_combo[0] for a in action_combo):
            reward_tensor[action_combo] = reward_matrix[action_combo[0], action_combo[0]]
        elif any(a == 0 for a in action_combo) and any(a == 2 for a in action_combo):
            reward_tensor[action_combo] = 0.0
        else:
            reward_tensor[action_combo] = 0.2

    return [reward_tensor] * n_players

def is_diagonal(matrix):
    return np.allclose(matrix, np.diag(np.diagonal(matrix)))

def normalizeMatrix(matrix, etendue):
    matrix_norm = (matrix-np.min(matrix))/np.ptp(matrix)
    matrix_norm_noise = matrix_norm*(1-etendue)+etendue/2
    return matrix_norm_noise