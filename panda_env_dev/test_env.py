import gym
import gym_panda

if __name__ == "__main__":
    done = False
    env = gym.make("panda-v0")
    # env = gym.make("PusherBulletEnv-v0")
    env.render()
    env.reset()
    while (not done):
        _, _, done, _ = env.step(env.action_space.sample())
