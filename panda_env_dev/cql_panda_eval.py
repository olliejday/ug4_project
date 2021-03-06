import os

import gym
from rlkit.samplers.data_collector import MdpPathCollector
import rlkit.torch.pytorch_util as ptu
import argparse
import torch
from rlkit.core import logger, eval_util
import matplotlib.pyplot as plt
import pandas as pd


def simulate_policy(fpath, env_name, seed, max_path_length,
                    num_eval_steps, headless, max_eps, verbose=True, pause=False):
    data = torch.load(fpath, map_location=ptu.device)
    policy = data['evaluation/policy']
    policy.to(ptu.device)
    # make new env, reloading with data['evaluation/env'] seems to make bug
    env = gym.make(env_name, **{"headless": headless, "verbose": False})
    env.seed(seed)
    if pause:
        input("Waiting to start.")
    path_collector = MdpPathCollector(env, policy)
    paths = path_collector.collect_new_paths(
                    max_path_length,
                    num_eval_steps,
                    discard_incomplete_paths=True,
                )

    if max_eps:
        paths = paths[:max_eps]
    if verbose:
        completions = sum([info["completed"] for path in paths for info in path["env_infos"]])
        print("Completed {} out of {}".format(completions, len(paths)))
        # plt.plot(paths[0]["actions"])
        # plt.show()
        # plt.plot(paths[2]["observations"])
        # plt.show()
        logger.record_dict(
            eval_util.get_generic_path_information(paths),
            prefix="evaluation/",
        )
        logger.dump_tabular()
    return paths


def plot_training(file):
    fpath = os.path.join(os.path.curdir, file)
    dir_path = os.path.dirname(fpath)
    if not fpath.endswith("progress.csv"):
        # then we have a directory path
        dir_path = fpath
        fpath = os.path.join(os.path.curdir, file, "progress.csv")
    rtns_cols = ["evaluation/Returns Mean", "evaluation/Returns Min", "evaluation/Returns Max"]
    epochs_col = "Epoch"
    df = pd.read_csv(fpath, index_col=epochs_col)
    print("Top mean returns")
    print(df.sort_values(rtns_cols, ascending=False)[rtns_cols][:15])
    df.plot(y=rtns_cols[0])
    plt.show()
    with open(os.path.join(dir_path, "best_models.txt"), "w") as f:
        f.write(str(df.sort_values(rtns_cols, ascending=False)[rtns_cols][:15]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=700,
                        help='Max length of rollout')
    parser.add_argument('--num_eval_steps', type=int, default=700 * 19,
                        help='Total number of eval steps to run')
    parser.add_argument('--env', type=str, default="panda-v0",
                        help='Gym env name')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--seed', default=14124, type=int)
    parser.add_argument('--max_eps', default=0, type=int, help="If limit number of total episodes")
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

        simulate_policy(args.file, args.env, args.seed, args.max_path_length,
                    args.num_eval_steps, args.headless, args.max_eps, pause=args.pause)
