import numpy as np
from execute import Execute
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl

def run_one_game_experiments(game_name, matrices, noise_levels, algos, rounds=500, horizon=1000, n_agents=2):
    results = {}
    for noise in tqdm(noise_levels):
        for algo_pair in algos:
            title = f"{algo_pair[0]}×{algo_pair[1]}_{noise[1]}"
            res = Execute(
                rounds,
                horizon,
                n_agents,
                [None] * n_agents,
                game_name
            ).getPDResult(
                matrices,
                algo_pair,
                'normal',
                noise
            )
            results[title] = res
    return results


def plot_results(game_name, results, noise_levels, algos, save_folder):
    mpl.rcParams["font.size"] = 5
    mpl.rcParams["axes.titlesize"] = 5
    mpl.rcParams["axes.labelsize"] = 5
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["ytick.labelsize"] = 5
    mpl.rcParams["legend.fontsize"] = 5
    sharey = True

    fig, axes = plt.subplots(1, 3, sharey=sharey, figsize=(5.2, 1.4), constrained_layout=False)
    fig.subplots_adjust(left=0,
                        right=1,
                        top=0.88,
                        bottom=0.15,
                        wspace=0.13)

    all_y_values = []
    for ax, noise in zip(axes, noise_levels):
        for algo_paired in algos:
            title = f"{algo_paired[0]}×{algo_paired[1]}_{noise[1]}"
            mean = results[title]['metrics']['mean_cum_regret']['agent_0']
            std = results[title]['metrics']['std_cum_regret']['agent_0']

            n_rounds = len(mean)
            x = np.arange(n_rounds)
            line = ax.plot(
                x,
                mean,
                label=f"{algo_paired[0]}$\\times${algo_paired[1]}",
                linewidth=1,
            )
            ax.plot(
                x,
                mean + std,
                linestyle='--',
                alpha=0.5,
                color=line[0].get_color(),
                linewidth=1,
            )
        for line in ax.get_lines():
            all_y_values.extend(line.get_ydata())

    y_max = max(all_y_values)
    margin = 0.1 * (y_max - 0)
    y_max += margin

    for ax, noise in zip(axes, noise_levels):
        ax.set_xlim(0, n_rounds)
        ax.set_ylim(0, y_max)
        ax.set_title(f"$\\sigma_{{noise}}={noise[1]}$", pad=4)
        ax.set_xlabel("Round (t)", labelpad=2)

        ax.tick_params(
            axis="y",
            which="both",
            direction="out",
            pad=4,
            length=0,
            right=False,
            left=True
        )
        ax.tick_params(
            axis="x",
            which="both",
            direction="out",
            pad=2,
            length=0,
            top=False,
            bottom=True
        )

        ax.grid(alpha=0.3, linewidth=0.5)
        sns.despine(ax=ax, trim=True)
        ax.tick_params(axis='y', labelleft=True)

    axes[0].set_ylabel(
        "Mean cumulative regret $R(t)$",
        fontsize=5,
        labelpad=6
    )
    sns.set_style("whitegrid")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="upper left",
        handlelength=1.0,
        columnspacing=0.6,
        borderaxespad=0.4,
        bbox_to_anchor=(0.02, 0.98)
    )
    fig.savefig(f"{save_folder}/{game_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_proportions(game_name, results, noise_levels, algos, save_folder):
    mpl.rcParams["font.size"] = 5
    mpl.rcParams["axes.titlesize"] = 5
    mpl.rcParams["axes.labelsize"] = 5
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["ytick.labelsize"] = 5
    mpl.rcParams["legend.fontsize"] = 5

    sns.set_style("whitegrid")
    for noise in noise_levels:
        fig, axes = plt.subplots(
            1, 2,
            sharex=True, sharey=True,
            figsize=(5.3, 1.0),
            constrained_layout=True,
            dpi=300
        )
        all_y_values = []
        for ax, (a1, a2) in zip(axes, algos):
            title = f"{a1}×{a2}_{noise[1]}"
            counts = results[title]["metrics"]["vecteur_de_props"]
            n_rounds, n_codes = counts.shape
            x = np.arange(n_rounds)

            k = int(np.sqrt(n_codes))
            if k * k == n_codes:
                labels = [f"({i},{j})" for i in range(k) for j in range(k)]
            else:
                labels = [f"pair {i}" for i in range(n_codes)]

            for code in range(n_codes):
                ax.plot(
                    x, counts[:, code],
                    label=labels[code],
                    linewidth=1,
                    alpha=0.75,
                )
                all_y_values.extend(counts[:, code])
            ax.set_title(f"{a1}$\\times${a2}", pad=2)
            ax.tick_params(axis="x", direction="out", pad=2, length=4, top=False)
            ax.tick_params(axis="y", direction="out", pad=4, length=4, right=False)

            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
            ax.grid(
                color="gray",
                linestyle="-",
                linewidth=0.5,
                alpha=0.3
            )

        y_max = np.max(all_y_values)
        y_top = 1.1 * y_max

        for ax in axes:
            ax.set_xlabel("Round (t)", labelpad=2)
            ax.set_xlim(left=0, right=n_rounds - 1)
            ax.set_ylim(0, y_top)
            ax.margins(x=0, y=0)

        label = []
        handles, labs = axes[1].get_legend_handles_labels()
        label.extend(labs)
        updated_label = []
        for lab in label:
            x_str, y_str = lab.strip('()').split(',')
            x = int(x_str.strip()) + 1
            y = int(y_str.strip()) + 1
            updated_label.append(f'({x},{y})')

        fig.legend(
            handles,
            updated_label,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=9,
            frameon=True,
            facecolor="white",
            edgecolor="none"
        )
        fig.supylabel("Proportion over 500 runs")
        fig.subplots_adjust(wspace=0.25, bottom=0.25)
        filename = f"{save_folder}/compareProps_{game_name}_noise{noise[1]:.2f}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")

def plot_PDproportions(game_name, results, noise_levels, save_folder):
    mpl.rcParams["font.size"] = 5
    mpl.rcParams["axes.titlesize"] = 5
    mpl.rcParams["axes.labelsize"] = 5
    mpl.rcParams["xtick.labelsize"] = 5
    mpl.rcParams["ytick.labelsize"] = 5
    mpl.rcParams["legend.fontsize"] = 5

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(
        1, 2,
        sharex=True, sharey=True,
        figsize=(5.3, 1.0),
        constrained_layout=True,
        dpi=300
    )
    all_y_values = []
    for ax, noise in zip(axes, noise_levels):
        title = f"KLUCB×KLUCB_{noise[1]}"
        counts = results[title]["metrics"]["vecteur_de_props"]
        n_rounds, n_codes = counts.shape
        x = np.arange(n_rounds)

        k = int(np.sqrt(n_codes))
        if k * k == n_codes:
            labels = [f"({i},{j})" for i in range(k) for j in range(k)]
        else:
            labels = [f"pair {i}" for i in range(n_codes)]

        for code in range(n_codes):
            ax.plot(
                x, counts[:, code],
                label=labels[code],
                linewidth=1,
                alpha=0.75,
            )
            all_y_values.extend(counts[:, code])
        ax.set_title(f"KLUCB$\\times$KLUCB", pad=2)
        ax.tick_params(axis="x", direction="out", pad=2, length=4, top=False)
        ax.tick_params(axis="y", direction="out", pad=4, length=4, right=False)

        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.grid(
            color="gray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.3
        )

        y_max = np.max(all_y_values)
        y_top = 1.1 * y_max

    for ax in axes:
        ax.set_xlabel("Round (t)", labelpad=2)
        ax.set_xlim(left=0, right=n_rounds - 1)
        ax.set_ylim(0, y_top)
        ax.margins(x=0, y=0)

    label = []
    handles, labs = axes[1].get_legend_handles_labels()
    label.extend(labs)
    updated_label = []
    for lab in label:
        x_str, y_str = lab.strip('()').split(',')
        x = int(x_str.strip()) + 1
        y = int(y_str.strip()) + 1
        updated_label.append(f'({x},{y})')

    fig.legend(
        handles,
        updated_label,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=9,
        frameon=True,
        facecolor="white",
        edgecolor="none"
    )
    fig.supylabel("Proportion over 500 runs")
    fig.subplots_adjust(wspace=0.25, bottom=0.25)
    filename = f"{save_folder}/compareProps_{game_name}.pdf"
    fig.savefig(filename, dpi=300, bbox_inches="tight")

def plot_results_action(game_name, results, noise_levels, algos, save_folder):
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.titlesize"] = 8
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["legend.fontsize"] = 8
    all_y_values = []
    for noise in noise_levels:
        fig, axes = plt.subplots(
            len(algos), 1,
            sharex=True, sharey=True,
            figsize=(5.7, 8),
            constrained_layout=False,
            dpi=300
        )
        for ax, (a1, a2) in zip(axes, algos):
            title = f"{a1}×{a2}_{noise[1]}"
            counts = results[title]["metrics"]["vecteur_de_props"]
            x = np.arange(counts.shape[0])

            n_codes = counts.shape[1]
            k = int(np.sqrt(n_codes))
            if k*k == n_codes:
                labels = [f"({i},{j})" for i in range(k) for j in range(k)]
            else:
                labels = [f"pair {i}" for i in range(n_codes)]

            for code in range(n_codes):
                ax.plot(
                    x, counts[:, code],
                    label=labels[code],
                    linewidth=1,
                    alpha=0.75,
                )
            for line in ax.get_lines():
                all_y_values.extend(line.get_ydata())
            ax.set_title(f"{a1}$\\times${a2}", pad=1)

        y_max = max(all_y_values)
        margin = 0.1 * (y_max - 0)
        y_max += margin

        for ax in axes:
            ax.grid(alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(0, y_max)
            ax.margins(x=0, y=0.05)
            sns.despine(ax=ax, trim=True)

        axes[-1].set_xlabel("Round (t)", labelpad=4)
        fig.supylabel("Proportion over 500 runs")

        label = []
        handles, labs = axes[1].get_legend_handles_labels()
        label.extend(labs)
        updated_label = []
        for lab in label:
            x_str, y_str = lab.strip('()').split(',')
            x = int(x_str.strip()) + 1
            y = int(y_str.strip()) + 1
            updated_label.append(f'({x},{y})')
        if game_name in ("PG", "PG_wp", "CG_no"):
            legendHeight = -0.04
        else:
            legendHeight = 0.0
        fig.legend(
            handles, updated_label,
            loc='lower center',
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, legendHeight),
        )
        fig.subplots_adjust(hspace=0.4)
        filename = f"{save_folder}/proportions_{game_name}_noise{noise[1]:.2f}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    np.random.seed(43)

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "figure.dpi": 300
    })
    sns.set_theme(style="whitegrid", palette="colorblind")

    games = {
        "PG_wp": [np.array([[1, 0.2, 0], [0.2, 0.8, 0.2], [0, 0.2, 1]]),
                  np.array([[1, 0.2, 0], [0.2, 0.8, 0.2], [0, 0.2, 1]])],
        "PG": [np.array([[1, 0, 0], [0, 0.2, 0], [0, 0, 1]]),
               np.array([[1, 0, 0], [0, 0.2, 0], [0, 0, 1]])],
        "PD": [np.array([[0.6, 0], [1, 0.4]]),
               np.array([[0.6, 0], [1, 0.4]]).T],
        "SG": [np.array([[1, 0], [0, 0.5]]),
               np.array([[1, 0], [0, 0.5]])],
        "CG_no": [np.array([[1, 0, 0.75], [0, 0.9, 0.85], [0.75, 0.75, 0.85]]),
                  np.array([[1, 0, 0.75], [0, 0.9, 0.85], [0.75, 0.75, 0.85]])]
    }

    noise_levels = [[0.0, 0.0], [0.0, 0.1], [0.0, 1.0]]
    algo_pairs = [
        ["UCB", "UCB"],
        ["KLUCB", "KLUCB"],
        ["TS", "TS"],
        ["UCB", "KLUCB"],
        ["UCB", "TS"]
    ]

    for game in tqdm(games.keys()):
        results = run_one_game_experiments(game ,games[game], noise_levels, algo_pairs, rounds=500, horizon=1000, n_agents=2)
        plot_results(game, results, noise_levels, algo_pairs, save_folder="Workshop/Figures")
        plot_results_action(game, results, noise_levels, algo_pairs, save_folder="Workshop/Figures")

    games_props = {
        "PG_wp": [np.array([[1, 0.2, 0], [0.2, 0.8, 0.2], [0, 0.2, 1]]),
                  np.array([[1, 0.2, 0], [0.2, 0.8, 0.2], [0, 0.2, 1]])],
        "SG": [np.array([[1, 0], [0, 0.5]]),
               np.array([[1, 0], [0, 0.5]])],
    }

    noise_levels_props = [[0.0, 0.0]]
    algo_pairs_props = [
        ["KLUCB", "KLUCB"],
        ["UCB", "KLUCB"],
    ]

    for game in tqdm(games_props.keys()):
        results = run_one_game_experiments(game, games_props[game], noise_levels_props, algo_pairs_props, rounds=500, horizon=1000,
                                           n_agents=2)
        plot_proportions(game, results, noise_levels_props, algo_pairs_props, save_folder="Workshop/FiguresCompare")

    PD = [np.array([[0.6, 0], [1, 0.4]]),
               np.array([[0.6, 0], [1, 0.4]]).T]
    noise = [[0.0, 0.0], [0.0, 0.1]]
    resultsPD = run_one_game_experiments("PD", PD, noise, [["KLUCB", "KLUCB"]], rounds=500,
                                       horizon=1000,
                                       n_agents=2)
    plot_PDproportions('PD', resultsPD, noise, save_folder="Workshop/FiguresCompare")
