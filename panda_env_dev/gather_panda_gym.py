import gym
import gym_panda
from panda_env_dev.pd_agent import PDAgent
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
"""


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }


def append_data(data, s, a, r, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())


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
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = gym.make("panda-v0")
    s = env.reset()
    done = False

    # Load the policy
    pd = PDAgent()
    pd.episode_start()

    data = reset_data()

    if args.video:
        frames = []

    ts = 0
    num_episodes = 0
    for _ in range(args.num_samples):
        act = pd.get_action(s)

        if args.noisy:
            act = act + np.random.randn(*act.shape) * 0.2
            act = np.clip(act, -1.0, 1.0)

        ns, r, done, info = env.step(act)

        pd.update_info(info)

        if ts >= args.max_episode_steps:
            done = True

        append_data(data, s[:-2], act, r, env.target_goal, done, env.physics.data)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1

        if done:
            done = False
            ts = 0
            s = env.reset()
            pd.episode_start()
            if args.video:
                frames = np.array(frames)
                save_video('./videos/', args.env + '_navigation', frames, num_episodes)

            num_episodes += 1
            frames = []
        else:
            s = ns

        if args.video:
            curr_frame = env.render()
            frames.append(curr_frame)
        elif args.render:
            env.render()


    fname = args.env + 'gym_panda_pd_agent.hdf5'
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == '__main__':
    main()
