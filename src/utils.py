import numpy as np

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