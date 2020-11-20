import gym
import gym_panda


class PDAgent:
    def __init__(self):
        self.error = 0.01
        self.fingers = 1
        self.object_position = [0.7, 0, 0.1]
        self.k_p = 10
        self.k_d = 1
        self.dt = 1. / 240.  # the default timestep in pybullet is 240 Hz
        self.fingers = 1

    def episode_start(self):
        self.fingers = 1

    def get_action(self, observation):
        dx = self.object_position[0] - observation[0]
        dy = self.object_position[1] - observation[1]
        target_z = self.object_position[2]
        if (observation[3] + observation[4]) < self.error + 0.02 and self.fingers == 0:
            target_z = 0.5
        dz = target_z - observation[2]
        if abs(dx) < self.error and abs(dy) < self.error and abs(dz) < self.error:
            self.fingers = 0
        pd_x = self.k_p * dx + self.k_d * dx / self.dt
        pd_y = self.k_p * dy + self.k_d * dy / self.dt
        pd_z = self.k_p * dz + self.k_d * dz / self.dt
        return [pd_x, pd_y, pd_z, self.fingers]

    def update_info(self, info):
        self.object_position = info["object_position"]


if __name__ == "__main__":
    env = gym.make('panda-v0')
    env.reset()
    pd = PDAgent()

    for i_episode in range(20):
        done = False
        observation = env.reset()
        cum_reward = 0
        pd.episode_start()
        for t in range(100):
            env.render()
            action = pd.get_action(observation)
            observation, reward, done, info = env.step(action)
            pd.update_info(info)
            cum_reward += reward
            if done:
                break
        print("Episode finished. timesteps: {}, reward: {}".format(t + 1, cum_reward))
    env.close()