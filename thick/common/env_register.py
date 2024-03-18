import common
from gym_minigrid.wrappers import *

def make_env(mode, config):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = common.DMC(
            task, config.action_repeat, config.render_size, config.dmc_camera)
        env = common.NormalizeAction(env)
    elif suite == 'atari':
        env = common.Atari(
            task, config.action_repeat, config.render_size,
            config.atari_grayscale)
        env = common.OneHotAction(env)
    elif suite == 'crafter':
        assert config.action_repeat == 1
        outdir = logdir / 'crafter' if mode == 'train' else None
        reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
        env = common.Crafter(outdir, reward)
        env = common.OneHotAction(env)
    elif suite == 'Minigrid':
        from common.minigrid_utils import MiniGridRegister
        env = common.GymWrapper(ImgObsWrapper(RGBImgPartialObsWrapper(MiniGridRegister(task))))
        env = common.OneHotAction(env)
    elif suite == 'Minihack':
        from common.minihack_utils import MiniHackRegister
        env = common.SuccessWrapper(common.GymWrapper(MiniHackRegister(task), env_obs_key='pixel_crop'))
        env = common.OneHotAction(env)
    elif suite == 'MultiWorld':
        from common.multiworld_utils import MultiWorldRegister
        env = common.GymWrapper(MultiWorldRegister(task), env_obs_key='image_observation')
        env = common.NormalizeAction(env)
    elif suite == 'pinpad':
        from common.pinpad_env import PinPad
        env = PinPad(task)
        env = common.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env