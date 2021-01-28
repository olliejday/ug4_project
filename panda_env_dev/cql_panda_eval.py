import os

import gym
from rlkit.samplers.data_collector import MdpPathCollector
import rlkit.torch.pytorch_util as ptu
import argparse
import torch
from rlkit.core import logger, eval_util
import matplotlib.pyplot as plt
import pandas as pd


def simulate_policy(args):
    data = torch.load(args.file, map_location=ptu.device)
    policy = data['evaluation/policy']
    policy.to(ptu.device)
    # make new env, reloading with data['evaluation/env'] seems to make bug
    env = gym.make(args.env, **{"headless": args.headless, "verbose": True})
    env.seed(args.seed)
    if args.pause:
        input("Waiting to start.")
    path_collector = MdpPathCollector(env, policy)
    paths = path_collector.collect_new_paths(
                    args.max_path_length,
                    args.num_eval_steps,
                    discard_incomplete_paths=True,
                )
    # plt.plot(paths[0]["actions"])
    # plt.show()
    # plt.plot(paths[2]["observations"])
    # plt.show()
    logger.record_dict(
        eval_util.get_generic_path_information(paths),
        prefix="evaluation/",
    )
    logger.dump_tabular()


def plot_training(file):
    fpath = os.path.join(os.path.curdir, file)
    if not fpath.endswith("progress.csv"):
        fpath = os.path.join(os.path.curdir, file, "progress.csv")
    rtns_cols = ["evaluation/Returns Mean", "evaluation/Returns Min", "evaluation/Returns Max"]
    epochs_col = "Epoch"
    df = pd.read_csv(fpath, index_col=epochs_col)
    print("Top mean returns")
    print(df.sort_values(rtns_cols, ascending=False)[rtns_cols][:15])
    df.plot(y=rtns_cols[0])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=700,
                        help='Max length of rollout')
    parser.add_argument('--num_eval_steps', type=int, default=700 * 10,
                        help='Total number of eval steps to run')
    parser.add_argument('--env', type=str, default="panda-v0",
                        help='Gym env name')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--seed', default=14124, type=int)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    if args.plot:
        plot_training(args.file)
    else:
        gpu_str = "0"
        if not args.no_gpu:
            ptu.enable_gpus(gpu_str)
            ptu.set_gpu_mode(True)

        simulate_policy(args)
