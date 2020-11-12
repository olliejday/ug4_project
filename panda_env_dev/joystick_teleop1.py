import gym
import pygame
from inputs import get_gamepad
import gym_panda

"""
Axis    Value                   Control
0 ABS_X      Left stick up/down      End effector Y
1 ABS_Y      Left stick L/R          End effector X
2       Left trigger            Gripper open
3 ABS_RX      Right stick up/down     End effector Z
4 ABS_RY      Right stick L/R
5       Right trigger           Gripper close

"""

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()

# Set the width and height of the screen [width,height]
size = [250, 500]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Joystick Teleop")

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()

js = pygame.joystick.Joystick(0)


def plot_sticks(event, screen):
    for i in range(js.get_numaxes() // 3):
        h = screen.get_height()
        w = screen.get_width()
        # js is [-1 to 1]
        # screen space is [0, 0] top left to [w, h] bottom right
        # we plot as [0, 0] centre of screen
        ax_screen_x = js.get_axis(i*3) * (w/2) + w/2
        ax_screen_y = js.get_axis(i*3+1) * (h/2) + h/2
        pygame.draw.circle(screen, (0, 0, 0), (ax_screen_x, ax_screen_y), 10)


def plot_trigger(event, screen):
    h = screen.get_height()
    w = screen.get_width()
    if js.get_axis(2) > 0:
        pygame.draw.rect(screen, (255, 0, 0), [0, 0, w/2, h/2])
    if js.get_axis(5) > 0:
        pygame.draw.rect(screen, (0, 255, 0), [w/2, 0, w/2, h/2])


def filter_stick(v, eps=1e-1):
    # filters close to 0 stick values
    return v if abs(v) > eps else 0


def get_action(events, scale=10):
    """
    Action is [dx, dy, dz, fingers]
    """
    dx = - filter_stick(even) * scale
    dy = - filter_stick(js.get_axis(0)) * scale
    dz = - filter_stick(js.get_axis(3)) * scale
    fingers = 0
    if js.get_axis(5) > 0:
        fingers = -js.get_axis(5)
    elif js.get_axis(2) < 0:
        fingers = js.get_axis(2)

    action = [dx, dy, dz, fingers]
    print(action)
    return action


# -------- Main Program Loop -----------

_plot = False
env = gym.make("panda-v0")
env.render()
obs = env.reset()

done = False

while not done:
    # EVENT PROCESSING STEP
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    events = get_gamepad()

    if _plot:
        # DRAWING STEP
        # First, clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
        screen.fill(WHITE)

        plot_trigger(js, screen)

        plot_sticks(js, screen)

    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

    # ACTION SET

    action = get_action(js)
    obs, rew, done, _ = env.step(action)

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # Limit to 60 frames per second
    clock.tick(60)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
env.close()
pygame.quit()