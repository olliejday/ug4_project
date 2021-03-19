import re

import gym
import numpy as np
import rlkit.torch.pytorch_util as ptu
import argparse
import torch
from PIL import Image
import os
import ffmpeg

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
    dir_path = os.path.join(save_dir, file_name + '_episode_{}'.format(episode_id))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(dir_path, 'frame_{}.png'.format(i)))
    composite_video(dir_path)


def composite_video(dir_path):
    ffmpeg\
        .input(os.path.join(dir_path, 'frame_%d.png'), framerate=50)\
        .output(os.path.join(dir_path, 'comp.mp4'))\
        .run(overwrite_output=True)


def main(env_name, policy, exp_name, max_episode_steps, num_episodes, seed):
    if not os.path.exists("data/videos"):
        os.makedirs("data/videos")

    env = gym.make(env_name, **{"headless": True, "verbose": False})
    env.seed(seed)
    env.reset()

    returns = []
    total_steps = 0
    completed_eps = 0
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        ts = 0
        cum_rew = 0
        frames = []
        while not done:
            act = policy.get_action(obs)[0]

            obs, r, done, info = env.step(act)

            cum_rew += r
            ts += 1

            if done:  # if done here then it was by env not by args.max_steps
                completed_eps += 1

            if ts >= max_episode_steps:
                done = True

            # video
            curr_frame = env.render()
            frames.append(curr_frame)

            if done:
                total_steps += ts
                returns.append(cum_rew)
                # save video
                frames = np.array(frames)
                save_video('data/videos/', exp_name + "_" + env_name, frames, i)


def combine_videos(path="data/videos", doComp=False):
    dirs = [l for l in os.listdir(path) if os.path.isdir(os.path.join(path, l))]
    experiments = {}
    for d in dirs:
        if doComp: composite_video(os.path.join(path, d))
        # all envs end in v[0-9]
        match = re.findall(".*v[0-9]", d)[0]
        if match not in experiments:
            experiments[match] = []
        experiments[match].append(os.path.join(path, d, "comp.mp4"))
    for exp_name, fs in experiments.items():
        vids = [ffmpeg.input(f) for f in fs]
        print(vids)
        ffmpeg\
            .concat(*vids)\
            .output(os.path.join(path, exp_name + ".mp4")) \
            .run(overwrite_output=True)


if __name__ == '__main__':
    """
    For RL models only, to save PD model use gather_panda_gym
        be sure to set seed and best iter
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=3, help='Num samples to collect')  # about 3M steps
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--seed', type=int, default=8392)
    parser.add_argument('--SAC', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--combine', action='store_true')

    args = parser.parse_args()

    # from experiments plot
    envs = ["panda-v0", "pandaForce-v0", "pandaPerturbed-v0"]

    if args.combine:
        combine_videos()
    else:
        gpu_str = "0"
        if not args.no_gpu:
            ptu.enable_gpus(gpu_str)
            ptu.set_gpu_mode(True)

        if args.SAC:
            exp_name = "SAC"
            best_itr = 72
        else:
            exp_name = "CQL"
            best_itr = 27
        dir_path = os.path.join("data", "{}-offline-panda-runs".format(exp_name))
        # get a directory to load model in
        dir_path = os.path.join(dir_path,
                                [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))][0])
        dir_path = os.path.join(dir_path, "{}_offline_panda_runs".format(exp_name))
        model_path = os.path.join(dir_path, os.listdir(dir_path)[0], "itr_{}.pkl".format(best_itr))

        data = torch.load(model_path, map_location=ptu.device)
        policy = data['evaluation/policy']
        policy.to(ptu.device)
        for env in envs:
            main(env, policy, exp_name, args.max_episode_steps, args.num_episodes, args.seed)
