from gym.envs.registration import register

register(
    id='dynamic-pong-v1',
    entry_point='gym_dynamic_pong.envs:DynamicPongEnv',
    # timestep_limit=1000,
    # reward_threshold=10,
    nondeterministic=False,
)
