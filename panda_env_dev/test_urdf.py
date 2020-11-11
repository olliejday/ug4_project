import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.setAdditionalSearchPath(pd.getDataPath())

timeStep = 1. / 60.
p.setTimeStep(timeStep)
p.setGravity(0, -9.8, 0)

# p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
p.loadURDF("bh_alone.urdf", useFixedBase=True)

while (1):
    p.stepSimulation()
    time.sleep(timeStep)