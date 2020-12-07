import json
import time

import rlkit.torch.pytorch_util as ptu
import torch
from d4rl import get_keys
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import argparse, os
import numpy as np

import h5py
import d4rl
import gym
import pybullet as p
import gym_panda


def load_params(f):
    """
    ['trainer/policy', 'trainer/qf1', 'trainer/qf2', 'trainer/target_qf1',
    'trainer/target_qf2', 'exploration/env', 'evaluation/env', 'evaluation/policy']
    """
    data = torch.load(f)
    # p.connect(p.GUI)
    return data


def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    # Center reward for Ant-Maze
    replay_buffer._rewards = (np.expand_dims(dataset['rewards'], 1) - 0.5)*4.0   
    replay_buffer._terminals = np.expand_dims(dataset['terminals'], 1)  
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def get_dataset(h5path, env):
    observation_space = env.observation_space
    action_space = env.action_space
    dataset_file = h5py.File(h5path, 'r')
    data_dict = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
    dataset_file.close()

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if observation_space.shape is not None:
        assert data_dict['observations'].shape[1:] == observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (str(data_dict['observations'].shape[1:]), str(observation_space.shape))
    assert data_dict['actions'].shape[1:] == action_space.shape, \
                'Action shape does not match env: %s vs %s' % (str(data_dict['actions'].shape[1:]), str(action_space.shape))
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:,0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:,0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (str(data_dict['rewards'].shape))
    return data_dict


def experiment(variant, data):
    # make new env, reloading with data['evaluation/env'] seems to make bug
    eval_env = gym.make("panda-v0", **{"headless": args["headless"]})
    eval_env.seed(variant['seed'])
    expl_env = eval_env

    qf1 = data['trainer/qf1']
    qf2 = data['trainer/qf2']
    target_qf1 = data['trainer/target_qf1']
    target_qf2 = data['trainer/target_qf2']
    policy = data['trainer/policy']
    eval_policy = data["evaluation/policy"]
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    if variant['load_buffer'] and buffer_filename is not None:
        replay_buffer.load_buffer(buffer_filename)
    else:
        dataset = get_dataset(variant["h5path"], eval_env)
        load_hdf5(d4rl.qlearning_dataset(eval_env, dataset), replay_buffer)
       
    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=True,
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train(start_epoch=variant["start_epoch"])


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


def load_variant(exp_dir):
    variant_file = os.path.join(exp_dir, "variant.json")
    with open(variant_file) as json_file:
        variant = json.load(json_file)
    return variant

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str, help="Experiment directory to load params and append logs")
    parser.add_argument('start_epoch', type=int, help="Start epoch for continue training logs")
    parser.add_argument("--params_fname", default="params.pkl", type=str)
    parser.add_argument('--gui', action='store_true')

    args = parser.parse_args()

    variant = load_variant(args.exp_dir)
    variant["start_epoch"] = args.start_epoch
    variant['headless'] = not args.gui

    params_data = load_params(os.path.join(args.exp_dir, args.params_fname))
    setup_logger(log_dir=args.exp_dir,
                 variant=variant)

    experiment(variant, params_data)