import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import rlkit.torch.pytorch_util as ptu
import json

from pd_agent import run_pd_agent
from cql_panda_eval import simulate_policy


def get_training_file_df(fpath):
    epochs_col = "Epoch"
    df = pd.read_csv(fpath, index_col=epochs_col)
    return df


def plot_training(dir_path, model_name, color, n_epochs=100, colname="evaluation/Returns Mean"):
    dir_path = os.path.join(dir_path, "{}-offline-panda-runs".format(model_name))
    rtns_to_plot = []
    for f in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(os.path.curdir, dir_path, f)): continue
        fpath = os.path.join(os.path.curdir, dir_path, f, "{}_offline_panda_runs".format(model_name))
        fpath = os.path.join(fpath, os.listdir(fpath)[0], "progress.csv")
        df = get_training_file_df(fpath)
        rtns = df[colname].truncate(after=n_epochs)
        rtns_to_plot.append(rtns)
    print(model_name, " Argmax mean return (over seeds): ", np.argsort(-np.mean(rtns_to_plot, axis=0))[:5])
    plot_df = pd.concat(rtns_to_plot)
    sns.lineplot(x=plot_df.index, y=plot_df, color=color, label=model_name)


def main_plot(dir_path, n_epochs=100, colname="evaluation/Returns Mean"):
    """
    Plots all the models over training
    PD benchmark reward = 2000, run run_pd_agent_seeds to confirm
    """
    palette = sns.color_palette()
    plot_training(dir_path, "CQL", palette[0], n_epochs, colname)
    plot_training(dir_path, "SAC", palette[1], n_epochs, colname)
    # add pd benchmark
    plt.hlines(2000, 0, 100, linestyles="dashed", colors=palette[2], label="PD agent")
    plt.title("Mean and std deviation of episode returns during training")
    plt.legend(loc="best")
    plt.savefig(os.path.join(dir_path, "training_plot.pdf"))


def run_pd_agent_seeds(dir_path, env_name, seeds, max_path_length=1000, n_eps=100):
    """
    Runs pd agent on all `seeds` for 100 episodes
    Saves returns to pd dataframe and prints out summary stats
    """
    all_rtns = {}
    for seed in seeds:
        rtns = run_pd_agent(env_name, seed, n_eps, max_path_length)
        all_rtns[seed] = rtns[:100]
    df = pd.DataFrame(all_rtns)
    df.to_csv(os.path.join(dir_path, "pd_agent_eval-{}.csv".format(env_name)))
    print("PD, ", env_name)
    print(df.describe())


def run_rl(dir_path, flavour, itr, env_name, max_path_length=1000, n_eps=100):
    """
    `dir_path` path to data
    Runs `flavour` ie. SAC/CQL
    Of all experiment directories for that flavour, runs iteration `itr`
    In environment `env_name`
    """
    dir_path = os.path.join(dir_path, "{}-offline-panda-runs".format(flavour))
    all_rtns = {}
    for f in os.listdir(dir_path):
        if not os.path.isdir(os.path.join(os.path.curdir, dir_path, f)): continue
        fpath = os.path.join(os.path.curdir, dir_path, f, "{}_offline_panda_runs".format(flavour))
        json_path = os.path.join(fpath, os.listdir(fpath)[0], "variant.json".format(itr))
        with open(json_path) as f:
            seed = json.loads(f.read())["seed"]
        model_path = os.path.join(fpath, os.listdir(fpath)[0], "itr_{}.pkl".format(itr))
        paths = simulate_policy(model_path, env_name, seed, max_path_length,
                    max_path_length * n_eps + 10, True, n_eps, verbose=False, pause=False)
        returns = [sum(path["rewards"]) for path in paths]
        all_rtns[seed] = np.concatenate(returns[:100])
    df = pd.DataFrame(all_rtns)
    df.to_csv(os.path.join(dir_path, "{}_eval-{}.csv".format(flavour, env_name)))
    print(flavour, ", ", env_name)
    print(df.describe())


def main_eval(dir_path, env, sac_best_itr=72, cql_best_itr=27):
    """
    sac_best_itr and cql_best_itr are retrieved from the plots of the training
    evaluation returns (main_plot).
    """
    print("Running all models for ", env)
    run_pd_agent_seeds(dir_path, env, SEEDS)
    run_rl(dir_path, "SAC", sac_best_itr, env)
    run_rl(dir_path, "CQL", cql_best_itr, env)


if __name__ == "__main__":
    SEEDS = [117, 12321, 7456, 3426, 573]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default="data",
                        help='path to the snapshot file')
    parser.add_argument('--env', type=str, default="panda-v0",
                        help='environment to eval in')
    parser.add_argument('--max_path_length', type=int, default=700,
                        help='Max length of rollout')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    gpu_str = "0"
    if not args.no_gpu:
        ptu.enable_gpus(gpu_str)
        ptu.set_gpu_mode(True)

    # run_pd_agent_seeds(args.dir_path, args.env, SEEDS)
    # main_plot(args.dir_path)

    # TODO:
    run_rl(args.dir_path, "CQL", 27, "panda-v0")
    run_rl(args.dir_path, "CQL", 27, "pandaForce-v0")
    # for env in ["pandaPerturbed-v0"]:
    #     main_eval(args.dir_path, env)
