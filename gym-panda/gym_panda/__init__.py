from gym.envs.registration import register

register(
    id='panda-v0',
    entry_point='gym_panda.envs:PandaEnv',
)

register(
    id='pandaPerturbed-v0',
    entry_point='gym_panda.envs:PandaEnvPerturbed',
)

register(
    id='pandaForce-v0',
    entry_point='gym_panda.envs:PandaEnvForce',
)

register(
    id='pandaObject-v0',
    entry_point='gym_panda.envs:PandaEnvObject',
)


