"""

Control:

Mouse controls end effector x and y location
"W" key is up in Z and "S" key is down
"A" key opens the gripper, "D" key closes it

Note Tkinter (0, 0) is Top Left

"""
import time

try:
    import tkinter
except ImportError:
    import Tkinter as tkinter

import threading
import numpy as np


class MouseTeleop(threading.Thread):
    def __init__(self, gripper_pos, obj_pos):
        threading.Thread.__init__(self)
        self._frequency = 1000
        self.pos_gripper_actual_world = gripper_pos
        self.obj_pos_world = obj_pos
        # params:
        self._scale = 1.0
        self.setup()

        self.start()

    def place_gripper(self):
        self.rect_gripper_actual = self._canvas.create_rectangle(0, 0, 0, 0)
        # gripper will follow mouse
        self.rect_gripper_desired = self._canvas.create_rectangle(0, 0, 0, 0)

    def place_object(self, pos_tk):
        x = pos_tk[0]
        y = pos_tk[1]
        self.obj_circle = self._canvas.create_oval(x-10, y-10, x+10, y+10)

    def setup(self):
        # set in mouse callback
        self.pos_gripper_desired_tk = None
        self.width_world = 100  # TODO
        self.height_world = 100

    def configure(self, event):
        self._width, self._height = event.height, event.width

    def send_action(self):
        # get the difference between the actual and desired
        if self.pos_gripper_desired_tk is not None:
            pos_gripper_desired_world = self.tk_to_world(self.pos_gripper_desired_tk)
            dx = pos_gripper_desired_world[0] - self.pos_gripper_actual_world[0]
            dy = pos_gripper_desired_world[1] - self.pos_gripper_actual_world[1]
            return [dx, dy, 0, 0]
        return [0, 0, 0, 0]

    def poll(self):
        pos_gripper_actual_tk = self.world_to_tk(self.pos_gripper_actual_world)
        self.observation_callback(pos_gripper_actual_tk[0], pos_gripper_actual_tk[1])
        self._root.after(self._frequency, self.poll)

    def run(self):
        # Create window:
        self._root = tkinter.Tk()
        self._root.title('Mouse Teleop')

        # Make window non-resizable:
        self._root.resizable(0, 0)

        self._canvas = tkinter.Canvas(self._root)
        self.place_gripper()
        self.place_object(self.world_to_tk(self.obj_pos_world))
        self._canvas.bind('<Motion>', self.mouse_callback)
        self._canvas.bind('<Configure>', self.configure)
        self._canvas.pack()

        self.poll()
        self._root.mainloop()

    def mouse_callback(self, event):
        self.pos_gripper_desired_tk = (event.x, event.y)
        self._canvas.coords(self.rect_gripper_desired,
                            event.x - 10,
                            event.y - 10,
                            event.x + 10,
                            event.y + 10)

    def observation_callback(self, x_tk, y_tk):
        self._canvas.coords(self.rect_gripper_actual,
                            x_tk - 10,
                            y_tk - 10,
                            x_tk + 10,
                            y_tk + 10)

    def tk_to_world(self, pos_tk):
        pos_world = [0, 0]  # TODO
        return pos_world

    def world_to_tk(self, pos_world):
        pos_tk = [0, 0]  # TODO
        return pos_tk


def main():
    gripper_pos = [10, 10]
    obj_pos = [10, 100]
    MouseTeleop(gripper_pos, obj_pos)
    time.sleep(2)
    gripper_pos[0] = 100
    gripper_pos[1] = 100


if __name__ == '__main__':
    main()
