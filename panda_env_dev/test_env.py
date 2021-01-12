import gym
import gym_panda
import numpy as np


if __name__ == "__main__":
    done = False
    env = gym.make("panda-v0", **{"headless": False})
    # env = gym.make("PusherBulletEnv-v0")
    env.render()
    env.reset()
    i = 0
    while (not done):
        # ac = env.action_space.sample()
        ac = np.ones(4)
        ac[-1] = 0.001 * i
        print(ac)
        obs, _, done, _ = env.step(ac)
        i += 1