import gym
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger, eval_util
import pybullet as p
import gym_panda

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    # make new env, reloading with data['evaluation/env'] seems to make bug
    env = gym.make("panda-v0", **{"headless": args.headless})
    env.seed(args.seed)
    print("Policy loaded")
    if not args.no_gpu:
        set_gpu_mode(True)
        policy.cuda()
    paths = []
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,  # we use p.conenct(p.GUI) to visualise, render makes slower
        )
        paths.append(path)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        else:
            logger.record_dict(
                eval_util.get_generic_path_information(paths),
                prefix="evaluation/",
            )
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=1500,
                        help='Max length of rollout')
    parser.add_argument('--env', type=str, default="panda-v0",
                        help='Gym env name')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
