import time

import torch
import rlkit.torch.pytorch_util as ptu
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
import gym_panda


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


def experiment(variant):
    eval_env = gym.make(variant['env_name'], **{"headless": variant["headless"]})
    eval_env.seed(variant['seed'])
    expl_env = eval_env
    
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M], 
    )
    eval_policy = MakeDeterministic(policy)
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
    # TODO: remove with, once figured out the issue!
    with torch.autograd.set_detect_anomaly(True):
        algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        env_name='panda-v0',
        sparse_reward=False,
        h5path="data/gym_panda_pd_agent.hdf5",
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1000,
            max_path_length=2000,
            num_eval_steps_per_epoch=4000,
            num_expl_steps_per_train_loop=4000,
            min_num_steps_before_training=2000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=3e-5,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=20000,  # Defaulted to 20000 (40000 or 10000 work similarly)
            num_qs=2,

            # CQL
            temp=5.0,
            min_q_version=3,  # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho))
            min_q_weight=1.0,  # the value of alpha, set to 5.0 or 10.0 if not using lagrange

            # lagrange
            with_lagrange=True,   # Defaults to true
            lagrange_thresh=10.0,  # the value of tau, corresponds to the CQL(lagrange) version

            # extra params
            num_random=10,
            max_q_backup=False,  # if we want to try max_{a'} backups, set this to true
            deterministic_backup=True,  # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='panda-v0')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')

    args = parser.parse_args()

    variant['buffer_filename'] = None

    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed
    variant['headless'] = not args.gui

    gpu_str = "0"
    if not args.no_gpu:
        ptu.enable_gpus(gpu_str)
        ptu.set_gpu_mode(True)

    rnd = np.random.randint(0, 1000000)
    setup_logger(os.path.join('CQL_offline_panda_runs', str(time.time()).split(".")[0]),
                 variant=variant, base_log_dir='./data')

    experiment(variant)
