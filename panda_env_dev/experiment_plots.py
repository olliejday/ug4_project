import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pd_agent import run_pd_agent


def get_training_file_df(fpath):
    epochs_col = "Epoch"
    df = pd.read_csv(fpath, index_col=epochs_col)
    return df


def plot_training(dir_path, model_name, color, n_epochs=100, colname="evaluation/Returns Mean"):
    rtns_to_plot = []
    for f in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(os.path.curdir, dir_path, f)): continue
        fpath = os.path.join(os.path.curdir, dir_path, f, "CQL_offline_panda_runs")
        fpath = os.path.join(fpath, os.listdir(fpath)[0], "progress.csv")
        df = get_training_file_df(fpath)
        rtns = df[colname].truncate(after=n_epochs)
        rtns_to_plot.append(rtns)
    plot_df = pd.concat(rtns_to_plot)
    sns.lineplot(x=plot_df.index, y=plot_df, color=color, label=model_name)


def main_plot(dir_path, n_epochs=100, colname="evaluation/Returns Mean"):
    """
    Plots all the models over training
    PD benchmark reward = 2000, run run_pd_agent_seeds to confirm
    """
    palette = sns.color_palette()
    plot_training(dir_path, "CQL", palette[0], n_epochs, colname)
    # add pd benchmark
    plt.hlines(2000, 0, 100, linestyles="dashed", colors=palette[1], label="PD agent")
    plt.title("Mean and std deviation of episode returns during training")
    plt.legend(loc="best")
    plt.savefig(os.path.join(dir_path, "training_plot.pdf"))


def run_pd_agent_seeds(dir_path, env_name, seeds):
    """
    Runs pd agent on all `seeds` for 100 episodes
    Saves returns to pd dataframe and prints out summary stats
    """
    all_rtns = {}
    for seed in seeds:
        rtns = run_pd_agent(env_name, seed, 100, 500)
        all_rtns[seed] = rtns[:100]
    df = pd.DataFrame(all_rtns)
    df.to_csv(os.path.join(dir_path, "pd_agent.csv"))
    print(df.describe())


if __name__ == "__main__":
    SEEDS = [117, 12321, 7456, 3426, 573]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default="data/CQL-offline-panda-runs",
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=700,
                        help='Max length of rollout')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    env_name = "panda-v0"
    # run_pd_agent_seeds(args.dir_path, env_name, SEEDS)

    main_plot(args.dir_path)