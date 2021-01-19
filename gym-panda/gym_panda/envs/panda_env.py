import gym
from gym import error, spaces, utils
from gym.utils import seeding

from rlkit.core import logger

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import matplotlib.pyplot as plt

"""
Note this env has Y axis as up
"""

pandaJoints = ['panda_joint1',
               'panda_joint2',
               'panda_joint3',
               'panda_joint4',
               'panda_joint5',
               'panda_joint6',
               'panda_joint7',
               'panda_joint8',
               'panda_hand_joint',
               'panda_finger_joint1',
               'panda_finger_joint2',
               'panda_grasptarget_hand'  # end effector
               ]
pandaJointsDict = {k: i for i, k in enumerate(pandaJoints)}

pandaNumDofs = 9

MAX_EPISODE_LEN = 20 * 100

reward_weights = {
    "reward_dist": 1,  # keep as 1 for base unit (typically -0.4 to 0)
    "reward_contacts": 0.07,
    "penalty_collision": 0.09,
    "reward_grasp": 1,
    "reward_z": 7.5,
    "reward_completion": 12.5,
}


class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, headless, verbose=False, sparse_reward=False):
        self.sparse_reward = sparse_reward
        self.step_counter = 0
        if headless:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.timeStep = 1. / 200.
        p.setTimeStep(self.timeStep)

        p.setGravity(0, -9.8, 0)
        p.resetDebugVisualizerCamera(cameraDistance=.6, cameraYaw=15, cameraPitch=-40, cameraTargetPosition=[.7, 0, .1])

        # see notes for details of bounds and of acs and obs spaces
        self.n_actions = pandaNumDofs
        self.end_effector_index = pandaJointsDict["panda_grasptarget_hand"]
        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        self.observation_space = spaces.Box(np.array([0] * 22), np.array([1] * 22))
        # normalistion params set empirically from expert data (see notes)
        # scale the std deviation to include more data in the -1 to 1 range
        scale_std = 7
        self.acs_mean = np.array([0.02008761, 0.25486009, -0.01521539, -2.08170228, 0.00326891,
                                  2.33839231, 2.35234778, 0.0397479, 0.0397479])
        self.acs_std = np.array([0.1882528, 0.30223908, 0.1093747, 0.26050974, 0.05065062,
                                 0.21928752, 0.29453576, 0.03993519, 0.03993519]) * scale_std
        self.obs_mean = np.array([0.06948607920799513, 0.01064014045877533, 0.09525471551881459,
                                  0.03932066822522797, 0.010644994960273341, 0.09547623670666466,
                                  0.11729979856542928, 0.5434034595684055, 0.00044514429090377434,
                                  0.18606473090696793, 0.015090418513382066, 0.248192467157062,
                                  -0.015294364772409075, -2.105414401533755, 0.003735502292605632,
                                  2.355187771295793, 2.3521831777351863, 0.029676513100094094,
                                  0.027724920054124354, -0.02947791052178796, 0.0003754021206685438,
                                  0.06876493234153894])
        self.obs_std = np.array([0.06509164, 0.02973735, 0.09385008, 0.02916187, 0.02982581,
                                 0.09411771, 0.11683397, 0.07144959, 0.14798064, 0.09347103,
                                 0.16901112, 0.29806916, 0.10117057, 0.26681981, 0.04518137,
                                 0.19360735, 0.28129071, 0.0126499, 0.01449605, 0.05605754,
                                 0.03160347, 0.09398505]) * scale_std

        self._max_episode_steps = MAX_EPISODE_LEN
        # whether to print out eg. if complete task
        self.verbose = verbose
        self.completed = False

    def step(self, action):
        """
        :param action: joint position controls in action space (action bounds), then scaled to joint space
        """
        assert np.shape(action) == (self.n_actions,)
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        action = self.process_action(action)

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        forces = np.array([100] * 9)
        p.setJointMotorControlArray(self.pandaUid, list(range(7)) + [9, 10], p.POSITION_CONTROL,
                                    action,
                                    forces=forces)

        p.stepSimulation()

        self.observation, _ = self.get_obs()

        done = False
        done, reward, _ = self.get_reward(done)
        # done here is that we completed, if completed we stay completed until env.reset()
        self.completed = self.completed or done

        self.step_counter += 1

        if self.step_counter > self._max_episode_steps:
            reward = 0
            done = True

        info = {
            "obj_pos": np.array(p.getBasePositionAndOrientation(self.objectUid)[0]),
            "hand_pos": np.array(p.getLinkState(self.pandaUid, 11)[0]),
            "fingers_joint": np.array([p.getJointState(self.pandaUid, 9)[0],
                                       p.getJointState(self.pandaUid, 10)[0]]),
            "completed": self.completed,
        }
        if self.completed and self.verbose:
            logger.log("Completed!")
        return self.observation, reward, done, info

    def get_reward(self, done):
        if self.sparse_reward:
            return self.get_sparse_reward(done)
        return self.get_dense_reward(done)

    def get_dense_reward(self, done):
        fingertip_pos = np.array(p.getLinkState(self.pandaUid, pandaJointsDict["panda_grasptarget_hand"])[0])
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_object = np.array(state_object)
        reward_dist = - np.linalg.norm(fingertip_pos - state_object)  # take hand to object

        # number of contacts (any contact between robot and object)
        reward_contacts = get_reward_contacts(self.pandaUid, self.objectUid)
        # number of collisions (contact between outside of hand and object
        penalty_collision = get_penalty_collision(self.pandaUid, self.objectUid)

        # if we have a contact that's not a collision and fingers are closed = likely a good grasp
        # position of less that 0.02 is roughly closed gripper
        reward_grasp = 0
        if reward_contacts - penalty_collision >= 1 and \
                p.getJointState(self.pandaUid, pandaJointsDict["panda_finger_joint1"])[0] < 0.03 and \
                p.getJointState(self.pandaUid, pandaJointsDict["panda_finger_joint2"])[0] < 0.03:
            reward_grasp = 1

        # task complete reward
        obj_z = state_object[2]
        # reward z but filter out slight values
        reward_z = obj_z if obj_z > 0.05 else 0
        reward_completion = 0
        if obj_z > 0.4:
            done = True
            reward_completion = 25
        reward_dict = {
            "reward_dist": reward_dist,
            "reward_contacts": reward_contacts,
            "reward_grasp": reward_grasp,
            "penalty_collision": penalty_collision,
            "reward_z": reward_z,
            "reward_completion": reward_completion,
        }
        reward = 0
        for k, r in reward_dict.items():
            reward += r * reward_weights[k]
        # plot_reward(reward_dict, reward)
        return done, reward, reward_dict

    def get_sparse_reward(self, done):
        fingertip_pos = np.array(p.getLinkState(self.pandaUid, pandaJointsDict["panda_grasptarget_hand"])[0])
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_object = np.array(state_object)
        reward_dist = - np.linalg.norm(fingertip_pos - state_object)  # take hand to object

        # task complete reward
        obj_z = state_object[2]
        # reward z but filter out slight values
        reward_completion = 0
        if obj_z > 0.4:
            done = True
            reward_completion = 200
        reward_dict = {
            # "reward_dist": reward_dist,
            "reward_completion": reward_completion,
        }
        reward = 0
        for k, r in reward_dict.items():
            reward += r * reward_weights[k]
        # plot_reward(reward_dict, reward)
        return done, reward, reward_dict

    def get_obs(self):
        fingertip_pos = np.array(p.getLinkState(self.pandaUid, pandaJointsDict["panda_grasptarget_hand"])[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self.objectUid)
        obj_pos = np.array(obj_pos)
        rel_pos = fingertip_pos - obj_pos

        # q pos of contollable / dof joints
        qpos_joints = np.array((p.getJointStates(self.pandaUid, list(range(7)) + [9, 10])), dtype=object)[:, 0]

        finger_pos1 = p.getLinkState(self.pandaUid, pandaJointsDict["panda_finger_joint1"])[0]
        finger_pos2 = p.getLinkState(self.pandaUid, pandaJointsDict["panda_finger_joint2"])[0]
        finger_pos = np.array([finger_pos1, finger_pos2])
        dist_fingers = np.sqrt((obj_pos - finger_pos) ** 2).reshape([-1])

        obs_dict = {
            "dist_fingers": dist_fingers,
            "obj_z": [obj_pos[2]],
            "palm_pos": fingertip_pos,
            "qpos_joints": qpos_joints,
            "rel_pos": rel_pos,
        }

        observation = np.concatenate([v for _, v in sorted(obs_dict.items())])
        observation = self.process_observation(observation)
        return observation, obs_dict

    def reset(self):
        self.completed = False
        self.step_counter = 0

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything
        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True)
        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid, 10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"), basePosition=[0.65, 0, 0])

        state_object = [random.uniform(0.5, 0.7), random.uniform(-0.2, 0.2), 0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/021/021.urdf"), basePosition=state_object,
                                    baseOrientation=[0, 0.5, 0, 0.5])

        # state_object = np.array(state_object) + np.random.uniform(0.05, 0.1, 3) * np.random.choice([-1, 1])
        # secondObject = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/002/002.urdf"), basePosition=state_object)
        self.observation, _ = self.get_obs()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return self.observation

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()

    def seed(self, seed=None):
        seed = seeding.create_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def process_action(self, action):
        # here take normalised actions 0 mean, 1 std and scales to action (joint) space (inverse of normalisation)
        return np.array(action, np.float) * self.acs_std + self.acs_mean

    def process_observation(self, obs):
        # from normal obs -> outputs 0 mean, 1 std observations
        return (np.array(obs, np.float) - self.obs_mean) / self.obs_std


def vector_angle_2d(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.arccos(x.dot(y)) * (180 / np.pi)


def get_penalty_collision(pandaUid, objectUid):
    """
    Collision taken as contact with outside of finger
    Penalty so negative
    """
    collisions = []
    for contact in p.getContactPoints():
        if contact[1] == pandaUid and contact[2] == objectUid:  # get robot and object contacts
            robot_is_B = 0  # robot is "B" in collision reported
        elif contact[2] == pandaUid and contact[1] == objectUid:
            robot_is_B = 1
        else:
            continue
        # collision if contact normal is opposite direction to the vector between the fingers
        contact_normal = np.array(contact[7])  # from B to A
        if not robot_is_B:  # want robot to object vector so reverse
            contact_normal *= -1
        finger_pos1 = np.array(p.getLinkState(pandaUid, pandaJointsDict["panda_finger_joint1"])[0])
        finger_pos2 = np.array(p.getLinkState(pandaUid, pandaJointsDict["panda_finger_joint2"])[0])
        finger_vector12 = finger_pos2 - finger_pos1  # vector from 1 to 2
        # normal vector angles used to compute collision or good contact, 90 deg + 10 deg leeway
        angle = vector_angle_2d(finger_vector12[:2], contact_normal[:2])
        # in xy a collision if finger 1 contact normal is opposite to finger (1 to 2) vector
        if contact[3 + robot_is_B] == pandaJointsDict["panda_finger_joint1"]:
            if angle >= 100 and angle <= 260:
                # print(1, finger_vector12, contact_normal, angle)
                collisions.append(contact)
        # in xy a collision if finger 2 contact normal is the same to finger (1 to 2) vector
        elif contact[3 + robot_is_B] == pandaJointsDict["panda_finger_joint2"]:
            if angle <= 80 or angle >= 280:
                # print(2, finger_vector12, contact_normal, angle)
                collisions.append(contact)
    return - len(collisions)


def get_reward_contacts(pandaUid, objectUid):
    """
    Number of contact points
    """
    contacts = []
    palm_contact = 0
    for contact in p.getContactPoints():
        if contact[1] == pandaUid and contact[2] == objectUid or \
                contact[2] == pandaUid and contact[1] == objectUid:  # get robot and object contacts
            contacts.append(contact)
            if contact[3] == pandaJointsDict["panda_hand_joint"] or contact[4] == pandaJointsDict[
                "panda_hand_joint"]:  # contact with palm
                palm_contact = 2
    return len(contacts) + palm_contact


rewards = {}


def plot_reward(reward_dict, reward):
    plt.clf()
    total = "total"
    if total in rewards:
        rewards[total].append(reward)
    else:
        rewards[total] = [reward]
    plt.plot(rewards[total], label=total)
    for k, v in reward_dict.items():
        if k in rewards:
            rewards[k].append(v * reward_weights[k])
        else:
            rewards[k] = [v * reward_weights[k]]
        plt.plot(rewards[k], label=k)
    plt.legend(loc="best")
    plt.draw()
    plt.pause(0.001)
