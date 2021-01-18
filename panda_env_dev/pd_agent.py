import argparse
import math

import pybullet as p
import gym
import gym_panda
import numpy as np
import matplotlib.pyplot as plt


class PDAgent:
    def __init__(self, env):
        assert hasattr(env, "pandaUid"), "Must call env.reset() to setup sim before PDAgent"
        self.env = env
        self.error = [0.017, 0.01, 0.01, 0.05]
        self.k_p = 10
        self.k_d = 1
        self.dt = 1. / 200.
        self.fingers = 0.08  # open
        self.n_actions = env.n_actions
        self.end_effector_index = env.end_effector_index
        self.pandaUid = env.pandaUid
        self.ll = [-7] * self.n_actions
        # upper limits for null space (todo: set them to proper range)
        self.ul = [7] * self.n_actions
        # joint ranges for null space (todo: set them to proper range)
        self.jr = [7] * self.n_actions
        # restposes for null space
        jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.rp = jointPositions

    def process_action(self, ac):
        # normalise to mean 0 and std 1
        return (np.array(ac, np.float) - self.env.acs_mean) / self.env.acs_std

    def episode_start(self):
        self.fingers = 0.08  # open

    def get_action(self, info):
        if info is None:
            return self.process_action(self.rp)  # rest pose
        object_position = info["obj_pos"]
        hand_position = info["hand_pos"]
        fingers_position = info["fingers_joint"]

        dx = object_position[0] - hand_position[0]
        dy = object_position[1] - hand_position[1]
        target_z = object_position[2]
        if (fingers_position[0] + fingers_position[1]) < self.error[3] and self.fingers == 0.0:  # if gripped object
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
        # pd action
        action = [pd_x, pd_y, pd_z, self.fingers]
        action = np.clip(action, -1, 1)
        # action to delta
        orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
        dv = 0.05
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]
        # change current position
        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid, self.end_effector_index, newPosition,
                                                  orientation, self.ll, self.ul, self.jr, self.rp, maxNumIterations=5)[0:7]
        return self.process_action(list(jointPoses) + 2 * [fingers])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='Pybullet gui')
    args = parser.parse_args()

    seed = 14124

    env = gym.make('panda-v0', **{"headless": args.headless})
    env.seed(seed)
    env.reset()

    pd = PDAgent(env)

    _a = []
    _o = []
    completed = 0

    for i_episode in range(10):
        done = False
        info = None
        observation = env.reset()
        cum_reward = 0
        pd.episode_start()
        for t in range(500):
            # pd agent outputs full action space so scale to ac input space (0, 1)
            action = pd.get_action(info)
            observation, reward, done, info = env.step(action)
            if info["completed"]:
                completed += 1
            cum_reward += reward
            if done:
                break
            _a.append(action)
            _o.append(observation)
        # print("Episode finished. timesteps: {}, reward: {}".format(t + 1, cum_reward))
    ###
    print("Completed: {}".format(completed))
    print("action")
    print("_mean = " + repr(np.mean(_a, axis=0)))
    print("_std = " + repr(np.std(_a, axis=0)))
    print("_min = " + repr(np.min(_a, axis=0)))
    print("_max = " + repr(np.max(_a, axis=0)))
    print("max {} min {}".format(np.max(_a), np.min(_a)))
    print("obs")
    print("_mean = " + repr(np.mean(_o, axis=0)))
    print("_std = " + repr(np.std(np.array(_o, np.float), axis=0)))
    print("_min = " + repr(np.min(_o, axis=0)))
    print("_max = " + repr(np.max(_o, axis=0)))
    print("max {} min {}".format(np.max(_o), np.min(_o)))
    ###
    env.close()
