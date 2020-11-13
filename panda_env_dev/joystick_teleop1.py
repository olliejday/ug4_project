import gym
import pygame
from inputs import get_gamepad
import gym_panda

"""
Axis    Value                   Control
0 ABS_Y      Left stick up/down      End effector Y
1 ABS_X      Left stick L/R          End effector X
2 ABS_Z      Left trigger            Gripper open
3 ABS_RY      Right stick up/down    End effector Z
4 ABS_RX      Right stick L/R
5 ABS_RZ      Right trigger          Gripper close

"""

MAX_JS_AXIS = 32767


def filter_stick(v, eps=500):
    # filters close to 0 stick values
    return v if abs(v) > eps else 0


def get_action(events, scale=10):
    """
    Action is [dx, dy, dz, fingers]
    """
    dx, dy, dz, fingers = 0, 0, 0, 0
    for event in events:
        if event.ev_type == 'Absolute':
            if event.code == "ABS_Y":
                dx = - filter_stick(event.state) * scale / MAX_JS_AXIS
            if event.code == "ABS_X":
                dy = - filter_stick(event.state) * scale / MAX_JS_AXIS
            if event.code == "ABS_RY":
                dz = - filter_stick(event.state) * scale / MAX_JS_AXIS
            fingers = 0
            if event.code == "ABS_Z":
                if event.state > 0:
                    fingers = -event.state / MAX_JS_AXIS
            if event.code == "ABS_RZ":
                if event.state < 0:
                    fingers = event.state / MAX_JS_AXIS

    action = [dx, dy, dz, fingers]
    print(action)
    return action


def main():
    global env
    env = gym.make("panda-v0")
    env.render()
    obs = env.reset()
    done = False
    while not done:
        events = get_gamepad()
        action = get_action(events)
        obs, rew, done, _ = env.step(action)
    env.close()


if __name__ == "__main__":
    main()
