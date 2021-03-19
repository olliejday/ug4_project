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


def plot_training(dir_path, model_name, color, n_epochs=100, colname="evaluation/Returns Mean",
                  scale=2000.0):
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
    plot_df /= scale
    sns.lineplot(x=plot_df.index, y=plot_df, color=color, label=model_name)


def plot_train_eval(dir_path, n_epochs=100, colname="evaluation/Returns Mean"):
    """
    Plots all the models over training
    PD benchmark reward = 2000, run run_pd_agent_seeds to confirm
    """
    # palette order to match eval plot
    palette = sns.color_palette()
    plt.hlines(1, 0, 100, linestyles="dashed", colors=palette[0], label="PD agent")
    plot_training(dir_path, "CQL", palette[4], n_epochs, colname)
    plot_training(dir_path, "SAC", palette[1], n_epochs, colname)
    # add pd benchmark
    plt.title("Evaluation returns during training (panda-v0)")
    plt.legend(loc="best")
    plt.savefig(os.path.join(dir_path, "training_plot.pdf"))


def plot_train_loss(dir_path, n_epochs=100, colnames=["trainer/Policy Loss"]):
    """
    trainer/Policy Loss
    trainer/Log Pis Mean
    trainer/QF1 Loss
    trainer/Q1 Predictions Mean

    Plots all the models over training
    PD benchmark reward = 2000, run run_pd_agent_seeds to confirm
    """
    # palette order to match eval plot
    palette = sns.color_palette()
    for colname in colnames:
        plot_training(dir_path, "CQL", palette[4], n_epochs, colname)
        # plot_training(dir_path, "SAC", palette[1], n_epochs, colname)
    # add pd benchmark
    plt.title("Policy Loss")
    plt.legend(loc="best")
    # plt.ylim(0, 1e6)
    plt.savefig(os.path.join(dir_path, "training_loss_plot.pdf"))


def plot_train_qf1(dir_path, n_epochs=100, colnames=["trainer/Q1 Predictions Mean"]):
    """
    trainer/Policy Loss
    trainer/Log Pis Mean
    trainer/QF1 Loss
    trainer/Q1 Predictions Mean

    Plots all the models over training
    PD benchmark reward = 2000, run run_pd_agent_seeds to confirm
    """
    # palette order to match eval plot
    palette = sns.color_palette()
    for colname in colnames:
        plot_training(dir_path, "CQL", palette[4], n_epochs, colname)
        # plot_training(dir_path, "SAC", palette[1], n_epochs, colname)
    # add pd benchmark
    plt.title("Q-function predictions")
    plt.legend(loc="best")
    # plt.ylim(0, 1e6)
    plt.savefig(os.path.join(dir_path, "training_QF1_plot.pdf"))


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


def get_eval_df(dir_path, model, env):
    if model == "PD":
        f_path = os.path.join(dir_path, "pd_agent_eval-{}.csv".format(env))
    else:
        f_path = os.path.join(dir_path, "{}-offline-panda-runs".format(model),
                              "{}_eval-{}.csv".format(model, env))
    df = pd.read_csv(f_path, index_col=0)
    return df


def plot_eval(dir_path, env_names):
    palette = sns.color_palette()
    pd_df = get_eval_plot_df(dir_path, "PD", env_names)
    cql_df = get_eval_plot_df(dir_path, "CQL", env_names)
    sac_df = get_eval_plot_df(dir_path, "SAC", env_names)
    plot_df = pd.concat([pd_df, cql_df, sac_df])
    plot_df.sort_index()
    mean_df = plot_df.groupby(["Model", "Environment"]).describe().loc[:, (slice(None), ['mean', 'std'])]
    mean_df = pd.DataFrame(mean_df.to_records())
    mean_df.columns = ["Model", "Environment", "Mean", "Std"]
    mean_df = mean_df.pivot("Model", "Environment").swaplevel(0, 1, axis=1)
    mean_df = mean_df.reindex(sorted(mean_df.columns), axis=1)
    print(mean_df.round(3).to_latex())

    plot_df.to_csv(os.path.join(dir_path, "eval_results.csv"))
    sns.catplot(data=plot_df, x="Environment", y="Mean Return", hue="Model", kind="bar",
                palette=[palette[0], palette[4], palette[1]])
    plt.title("100 episode returns of trained models")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "eval_plot.pdf"))


def get_eval_plot_df(dir_path, model, env_names):
    plot_means = []
    for i, env in enumerate(env_names):
        df = get_eval_df(dir_path, model, env)
        df = pd.DataFrame(df.mean() / 2000.0)
        df["env"] = env
        plot_means.append(df)
    plot_df = pd.concat(plot_means)
    plot_df.columns = ["Mean Return", "Environment"]
    # add model col
    plot_df["Model"] = [model] * len(plot_df)
    return plot_df


if __name__ == "__main__":
    SEEDS = [117, 12321, 7456, 3426, 573]
    envs = ["panda-v0", "pandaForce-v0", "pandaPerturbed-v0"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default="data",
                        help='path to the snapshot file')
    parser.add_argument('--env', type=str, default="panda-v0",
                        help='environment to eval in')
    parser.add_argument('--max_path_length', type=int, default=700,
                        help='Max length of rollout')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    # gpu_str = "0"
    # if not args.no_gpu:
    #     ptu.enable_gpus(gpu_str)
    #     ptu.set_gpu_mode(True)

    # run_pd_agent_seeds(args.dir_path, args.env, SEEDS)
    # plot_train_eval(args.dir_path)
    # plot_train_loss(args.dir_path)
    # plot_train_qf1(args.dir_path)

    # TODO:
    # for env in envs:
    #     main_eval(args.dir_path, env)

    plot_eval(args.dir_path, envs)
