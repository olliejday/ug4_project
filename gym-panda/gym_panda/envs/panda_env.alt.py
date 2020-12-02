import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

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
        p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        timeStep = 1. / 60.
        p.setTimeStep(timeStep)
        p.setGravity(0, -9.8, 0)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, cameraPitch=-40, cameraTargetPosition=[.5, 0, .5])
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*8), np.array([1]*8))
        self._max_episode_steps = MAX_EPISODE_LEN

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orn = p.getQuaternionFromEuler([math.pi/2.,0.,0.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.panda, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        jointPoses = p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, newPosition, orn, ll, ul, jr, rp, maxNumIterations=5)[0:7]

        for i in range(pandaNumDofs):
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
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
        state_robot = np.array(p.getLinkState(self.panda, 11)[0])
        state_object, _ = p.getBasePositionAndOrientation(self.sphereId)
        state_object = np.array(state_object)
        obs_object = state_robot - state_object
        state_fingers = np.array((p.getJointState(self.panda, 9)[0], p.getJointState(self.panda, 10)[0]))
        observation = np.concatenate([state_robot, state_fingers, obs_object])
        return observation, state_object, state_robot

    def reset(self):
        offset = [0, 0, 0]
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        legos = []
        p.loadURDF("tray/traybox.urdf", [0 + offset[0], 0 + offset[1], -0.6 + offset[2]],
                                    [-0.5, -0.5, -0.5, 0.5], flags=flags)
        legos.append(
            p.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.5]) + offset, flags=flags))
        legos.append(
            p.loadURDF("lego/lego.urdf", np.array([-0.1, 0.3, -0.5]) + offset, flags=flags))
        legos.append(
            p.loadURDF("lego/lego.urdf", np.array([0.1, 0.3, -0.7]) + offset, flags=flags))
        self.sphereId = p.loadURDF("random_urdfs/000/000.urdf", np.array([0, 0.3, -0.6]) + offset, flags=flags)
        # p.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.5]) + offset, flags=flags)
        # p.loadURDF("sphere_small.urdf", np.array([0, 0.3, -0.7]) + offset, flags=flags)
        orn = [-0.707107, 0.0, 0.0, 0.707107]  # p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        eul = p.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.panda = p.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]) + offset, orn,
                                                 useFixedBase=True, flags=flags)
        index = 0
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)

            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
        return self.get_obs()

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
