from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera_zoomed_fixed, \
  sawyer_pick_and_place_camera_zoomed_fixed2, sawyer_pick_and_place_camera_slanted_fixed, \
  sawyer_pick_and_place_camera_zoomed_fixed4, sawyer_pusher_camera_upright_v3
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv, SawyerPickAndPlaceEnvYZ
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEasyEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
import numpy as np


def MultiWorldRegister(name, action_repeat=1):

  if 'pickplaceyz' in name:
    base_env = SawyerPickAndPlaceEnvYZ(obj_init_positions=((0, 0.6, 0.016),), target=True)
    camera_view = sawyer_pick_and_place_camera_zoomed_fixed4
  elif 'push' in name:
    base_env = SawyerPushAndReachXYEasyEnv(fixed_target=True)
    camera_view = sawyer_pusher_camera_upright_v3
  elif 'doorhook' in name:
    base_env = SawyerDoorHookEnv(fix_goal=True, reset_free=True, use_puck=(name == 'doorhookpuck'))
    camera_view = sawyer_pick_and_place_camera_slanted_fixed
  else:
    raise ValueError(name)
  env = ImageEnv(
    base_env,
    transpose=False,
    flip_img=True,
    flatten_img=False,
    return_image_proprio=False,
    imsize=64,
    init_camera=camera_view,
  )
  env = MultiWorldRewardWrapper(env=env, name=name, action_repeat=action_repeat)
  return TimeLimit(env=env, duration=50)


class MultiWorldRewardWrapper:

  def __init__(self, env, name, action_repeat):
    self.multiworld_name = name
    self._env = env
    self._action_repeat = action_repeat
    self._t = 0

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def _overwrite_reward(self, reward):
    if self.multiworld_name == 'doorhook':
      reward = self._env.get_door_angle()
    elif 'push' in self.multiworld_name:
      init_dist = np.linalg.norm(self._env.get_target_xy() - self._env.sample_puck_xy())
      curr_dist = np.linalg.norm(self._env.get_target_xy() - self._env.get_puck_pos()[:2])
      if self.multiworld_name == 'pushsparse':
        reward = 1 if curr_dist < 0.025 else 0
      else:
        reward = 1 - (curr_dist/init_dist)
    elif 'pickplaceyz' in self.multiworld_name:
      start_dist = np.linalg.norm(self._env.obj_init_positions[0][1:] - self._env.get_target_pos()[1:])
      curr_dist = np.linalg.norm(self._env.get_obj_pos()[1:] - self._env.get_target_pos()[1:])
      reward = 1.0 - curr_dist/start_dist
    else:
      raise ValueError(self.multiworld_name)
    return reward

  def step(self, action):
    if 'pickplace' in self.multiworld_name:
      action[-1] = np.sign([action[-1] + 0.0001])  # binary grasp
    for _ in range(self._action_repeat):
      obs, reward, done, info = self._env.step(action)
      if done:
        break
    r = self._overwrite_reward(reward)
    self._t = self._t + 1
    return obs, r, done, info

  def reset(self):
    self._t = 0
    return self._env.reset()

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()

