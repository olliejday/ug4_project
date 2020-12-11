import gym
import gym_panda


class PDAgent:
    def __init__(self):
        self.error = [0.017, 0.01, 0.01]
        self.fingers = 1
        self.object_position = [0.7, 0, 0.1]
        self.k_p = 10
        self.k_d = 1
        self.dt = 1. / 60.  # the default timestep in pybullet is 240 Hz
        self.fingers = 1

    def episode_start(self):
        self.fingers = 1

    def get_action(self, observation):
        dx = self.object_position[0] - observation[0]
        dy = self.object_position[1] - observation[1]
        target_z = self.object_position[2]
        if (observation[3] + observation[4]) < self.error[2] + 0.025 and self.fingers == 0:  # if gripped object
            target_z = 0.5
        offset_z = 0.01
        dz = target_z - (observation[2] + offset_z)  # offset for better grip
        pd_x = self.k_p * dx + self.k_d * dx / self.dt
        pd_y = self.k_p * dy + self.k_d * dy / self.dt
        pd_z = self.k_p * dz + self.k_d * dz / self.dt
        if abs(dx) > self.error[0] * 3 or abs(dy) > self.error[1] * 3:  # get roughly over the object
            pd_z = 0
        if abs(dx) < self.error[0] and abs(dy) < self.error[1] and abs(dz) < self.error[2]:  # if gripper around object
            self.fingers = 0
        # print(abs(dx), observation[0], abs(dy), abs(dz))
        return [pd_x, pd_y, pd_z, self.fingers]

    def update_info(self, info):
        self.object_position = info["object_position"]


if __name__ == "__main__":
    seed = 1123

    env = gym.make('panda-v0', **{"headless": True})
    env.seed(seed)
    env.reset()
    pd = PDAgent()

    for i_episode in range(50):
        done = False
        observation = env.reset()
        cum_reward = 0
        pd.episode_start()
        for t in range(500):
            action = pd.get_action(observation)
            observation, reward, done, info = env.step(action)
            pd.update_info(info)
            cum_reward += reward
            if done:
                break
        print("Episode finished. timesteps: {}, reward: {}".format(t + 1, cum_reward))
    env.close()
