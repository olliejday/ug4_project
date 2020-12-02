import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


"""
Note this env has Y axis as up
"""

pandaEndEffectorIndex = 11 #8
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

    def __init__(self):
        self.step_counter = 0
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
        jointPoses = p.calculateInverseKinematics(self.pandaUid, pandaEndEffectorIndex, newPosition, orientation, ll, ul, jr, rp, maxNumIterations=5)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers],
                                    forces=[5 * 240.]*pandaNumDofs)

        p.stepSimulation()

        self.observation, state_object, state_robot = self.get_obs()

        done = False
        reward = -0.1 * np.linalg.norm(state_robot - state_object)  # take hand to object
        # task complete reward
        if state_object[2]>0.45:
            reward += 10.
            done = True

        self.step_counter += 1

        if self.step_counter > self._max_episode_steps:
            reward = 0
            done = True

        info = {'object_position': state_object}
        return self.observation.astype(np.float32), reward, done, info

    def get_obs(self):
        state_robot = np.array(p.getLinkState(self.pandaUid, 11)[0])
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_object = np.array(state_object)
        obs_object = state_robot - state_object
        state_fingers = np.array((p.getJointState(self.pandaUid, 9)[0], p.getJointState(self.pandaUid, 10)[0]))
        observation = np.concatenate([state_robot, state_fingers, obs_object])
        return observation, state_object, state_robot

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

        state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)

        # state_object = np.array(state_object) + np.random.uniform(0.05, 0.1, 3) * np.random.choice([-1, 1])
        # secondObject = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/002/002.urdf"), basePosition=state_object)
        self.observation, _, _ = self.get_obs()
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
