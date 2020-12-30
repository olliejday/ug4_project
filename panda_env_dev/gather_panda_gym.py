import gym
import gym_panda
from tqdm import tqdm

from pd_agent import PDAgent
import numpy as np
import pickle
import gzip
import h5py
import argparse
import torch
from PIL import Image
import os

"""
Uses PDAgent in gym_panda to gather an offline RL dataset

Saves relative to where run from to data/<exp_name>
If videos they save to videos/<exp_name><episode number> 

Note on Ubuntu for some reason it stops when the screen turns off
so disable automatic blank screen setting
"""


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/obj_pos': [],
            }


def append_data(data, s, a, r, obj, done):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['infos/obj_pos'].append(obj)


def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name + '_episode_{}'.format(episode_id))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=int(3e6), help='Num samples to collect')
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--gui', action='store_true', help='Pybullet gui')
    args = parser.parse_args()

    exp_name = "gym_panda_pd_agent"
    if not os.path.exists("data"):
        os.makedirs("data")

    env = gym.make("panda-v0", **{"headless": not args.gui})
    s = env.reset()
    info = None

    # Load the policy
    pd = PDAgent()
    pd.episode_start()

    data = reset_data()

    if args.video:
        frames = []

    ts = 0
    num_episodes = 0
    returns = []
    cum_rew = 0
    for _ in tqdm(range(args.num_samples)):
        act = pd.get_action(info)

        if args.noisy:
            act = act + np.random.randn(*act.shape) * 0.2
            act = np.clip(act, -1.0, 1.0)

        ns, r, done, info = env.step(act)

        if ts >= args.max_episode_steps:
            done = True

        append_data(data, s, act, r, info["object_position"], done)

        cum_rew += r
        ts += 1

        if done:
            ts = 0
            s = env.reset()
            pd.episode_start()
            if args.video:
                frames = np.array(frames)
                save_video('./videos/', exp_name, frames, num_episodes)

            num_episodes += 1
            returns.append(cum_rew)
            cum_rew = 0
            frames = []
        else:
            s = ns

        if args.video:
            curr_frame = env.render()
            frames.append(curr_frame)
        elif args.render:
            env.render()

    fname = exp_name + '.hdf5'
    if args.noisy:
        fname = exp_name + 'noisy.hdf5'
    fname = os.path.join("data", fname)
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

    print("Created dataset.")
    print("Saved to {}".format(fname))
    print("{} Episodes, {} mean return, {} max return, {} min return.".format(num_episodes,
                                                                              np.mean(returns),
                                                                              np.max(returns),
                                                                              np.min(returns)))



if __name__ == '__main__':
    main()
