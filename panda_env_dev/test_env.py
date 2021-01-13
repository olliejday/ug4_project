import gym
import gym_panda
import numpy as np


if __name__ == "__main__":
    done = False
    env = gym.make("panda-v0", **{"headless": False})
    env.render()
    env.reset()
    while (not done):
        ac = env.action_space.sample()
        obs, _, done, info = env.step(ac)
