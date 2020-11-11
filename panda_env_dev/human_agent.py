import gym
import gym_panda


def get_action():
    pass
    # TODO


if __name__ == "__main__":
    done = False
    env = gym.make("panda-v0")
    env.render()
    env.reset()
    while (not done):
        _, _, done, _ = env.step(get_action())
