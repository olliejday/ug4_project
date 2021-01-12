import argparse

import gym
import gym_panda
import numpy as np

class PDAgent:
    def __init__(self, acs_offset, acs_scale):
        self.error = [0.017, 0.01, 0.01, 0.035]
        self.k_p = 10
        self.k_d = 1
        self.dt = 1. / 60.  # the default timestep in pybullet is 240 Hz
        self.fingers = 0.08
        self.acs_offset = acs_offset
        self.acs_scale = acs_scale

    def process_action(self, ac):
        return (np.array(ac) - self.acs_offset) / self.acs_scale

    def episode_start(self):
        self.fingers = 0.08

    def get_action(self, info):
        if info is None:
            return self.process_action([0, 0, 0, self.fingers])
        object_position = info["obj_pos"]
        hand_position = info["hand_pos"]
        fingers_position = info["fingers_joint"]

        dx = object_position[0] - hand_position[0]
        dy = object_position[1] - hand_position[1]
        target_z = object_position[2]
        if (fingers_position[0] + fingers_position[1]) < self.error[3] and self.fingers == 0:  # if gripped object
            target_z = 0.5
        offset_z = 0.01
        dz = target_z - (hand_position[2] + offset_z)  # offset for better grip
        pd_x = self.k_p * dx + self.k_d * dx / self.dt
        pd_y = self.k_p * dy + self.k_d * dy / self.dt
        pd_z = self.k_p * dz + self.k_d * dz / self.dt

        if abs(dx) > self.error[0] * 3 or abs(dy) > self.error[1] * 3:  # get roughly over the object
            pd_z = 0
        if abs(dx) < self.error[0] and abs(dy) < self.error[1] and abs(dz) < self.error[2]:  # if gripper around object
            self.fingers = 0.0
        # print(abs(dx), observation[0], abs(dy), abs(dz))
        return self.process_action([pd_x, pd_y, pd_z, self.fingers])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Pybullet gui')
    args = parser.parse_args()

    seed = 14123

    env = gym.make('panda-v0', **{"headless": args.headless})
    env.seed(seed)
    env.reset()
    pd = PDAgent(env.acs_offset, env.acs_scale)

    _a = []
    _o = []

    for i_episode in range(100):
        done = False
        info = None
        observation = env.reset()
        cum_reward = 0
        pd.episode_start()
        for t in range(500):
            # pd agent outputs full action space so scale to ac input space (0, 1)
            action = pd.get_action(info)
            observation, reward, done, info = env.step(action)
            cum_reward += reward
            if done:
                break
            _a.append(action)
            _o.append(observation)
        # print("Episode finished. timesteps: {}, reward: {}".format(t + 1, cum_reward))
    ###
    print("action")
    print(np.max(_a, axis=0))
    print(np.min(_a, axis=0))
    print("obs")
    print(np.max(_o, axis=0))
    print(np.min(_o, axis=0))
    ###
    env.close()
