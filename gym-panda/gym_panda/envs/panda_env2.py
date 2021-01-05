import gym
from gym import error, spaces, utils
from gym.utils import seeding

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

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

MAX_EPISODE_LEN = 20*100

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, headless):
        self.reward_weights = {
            "reward_dist": 1,
            "reward_contacts": 1,
            "penalty_collision": 1,
            "reward_z": 3,
        }
        self.step_counter = 0
        if headless:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        timeStep = 1. / 60.
        p.setTimeStep(timeStep)
        p.setGravity(0, -9.8, 0)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, cameraPitch=-40, cameraTargetPosition=[.5, 0, .5])
        # TODO: set these (see notes.md)
        self.action_space = spaces.Box(np.array([-5]*4), np.array([5]*4))
        self.observation_space = spaces.Box(np.array([-1]*8), np.array([1]*8))
        self._max_episode_steps = MAX_EPISODE_LEN

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.pandaUid, pandaJointsDict["panda_grasptarget_hand"], newPosition, orientation, ll, ul, jr, rp, maxNumIterations=5)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers],
                                    forces=[5 * 240.]*pandaNumDofs)

        p.stepSimulation()

        self.observation, _ = self.get_obs()

        done = False
        done, reward, _ = self.get_reward(done)

        self.step_counter += 1

        if self.step_counter > self._max_episode_steps:
            reward = 0
            done = True

        info = {
            "obj_pos": np.array(p.getBasePositionAndOrientation(self.objectUid)[0]),
            "hand_pos": np.array(p.getLinkState(self.pandaUid, 11)[0]),
            "fingers_joint": np.array([p.getJointState(self.pandaUid, 9)[0],
                                     p.getJointState(self.pandaUid, 10)[0]])
        }
        return self.observation.astype(np.float32), reward, done, info

    def get_reward(self, done):
        state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_object = np.array(state_object)
        reward_dist = -0.1 * np.linalg.norm(state_robot - state_object)  # take hand to object

        reward_contacts = get_reward_contacts(self.pandaUid, self.objectUid)
        penalty_collision = get_penalty_collision(self.pandaUid, self.objectUid)

        # task complete reward
        reward_z = state_object[2]
        if state_object[2] > 0.4:
            done = True
        reward_dict = {
            "reward_dist": reward_dist,
            "reward_contacts": reward_contacts,
            "penalty_collision": penalty_collision,
            "reward_z": reward_z
        }
        reward = 0
        for k, v in self.reward_weights.items():
            reward += v * reward_dict[k]
        plot_reward(reward_dict)
        return done, reward, reward_dict

    def get_obs(self):
        hand_pos = np.array(p.getLinkState(self.pandaUid, pandaJointsDict["panda_grasptarget_hand"])[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self.objectUid)
        obj_pos = np.array(obj_pos)
        rel_pos = obj_pos - hand_pos

        fingers_joint_state = np.array((p.getJointState(self.pandaUid, pandaJointsDict["panda_finger_joint1"])[0],
                                        p.getJointState(self.pandaUid, pandaJointsDict["panda_finger_joint2"])[0]))

        finger_pos1 = p.getLinkState(self.pandaUid, pandaJointsDict["panda_finger_joint1"])[0]
        finger_pos2 = p.getLinkState(self.pandaUid, pandaJointsDict["panda_finger_joint2"])[0]
        finger_pos = np.array([finger_pos1, finger_pos2])
        dist_fingers = np.sqrt((obj_pos - finger_pos) ** 2).reshape([-1])

        joint_torques = np.concatenate([p.getJointState(self.pandaUid, i)[2] for i in range(len(pandaJointsDict))])

        obs_dict = {
            "rel_pos": rel_pos,
            "fingers_joint_state": fingers_joint_state,
            "dist_fingers": dist_fingers,
            "joint_torques": joint_torques
        }

        observation = np.concatenate([v for _, v in sorted(obs_dict.items())])
        return observation, obs_dict

    def reset(self):
        self.step_counter = 0
        # print(p.getPhysicsEngineParameters())
        # orig_params = {'fixedTimeStep': 0.016666666666666666, 'numSubSteps': 0, 'numSolverIterations': 50,
        #                'useRealTimeSimulation': 0, 'numNonContactInnerIterations': 1}
        # params = {'splitImpulsePenetrationThreshold': 1,}
        # p.setPhysicsEngineParameter(**params)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        state_object= [random.uniform(0.5,0.7),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/021/021.urdf"), basePosition=state_object,
                                    baseOrientation=[0, 0.5, 0, 0.5])

        # state_object = np.array(state_object) + np.random.uniform(0.05, 0.1, 3) * np.random.choice([-1, 1])
        # secondObject = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/002/002.urdf"), basePosition=state_object)
        self.observation, _ = self.get_obs()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return self.observation.astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

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


def plot_reward(reward, reward_dict):
    plt.clf()
    for k, v in reward_dict.items():
        if k in rewards:
            rewards[k].append(v)
        else:
            rewards[k] = [v]
        plt.plot(rewards[k], label=k)
    plt.legend(loc="best")
    plt.draw()
    plt.pause(0.001)
