import dynamic_gym.pong


class Env:
    raise NotImplementedError()


class Wrapper:
    raise NotImplementedError()


class TimeLimit:
    raise NotImplementedError()

class EnvSpec:
    raise NotImplementedError()

def make(version: str):
    spec = self.spec(path)
    env = spec.make(**kwargs)
    if env.spec.max_episode_steps is not None:
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)