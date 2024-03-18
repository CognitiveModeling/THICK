import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import wandb
from common.env_register import make_env

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common


def main(argv):

  configs = yaml.safe_load((
      pathlib.Path(argv[0]).parent / 'configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(argv=argv[1:], known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = common.Flags(config).parse(remaining)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')

  load_dir_str = config.load_dir
  loaddir = pathlib.Path(load_dir_str).expanduser()

  # WandB configs
  if config.use_wandb:
    os.environ["WANDB_API_KEY"] = config.wandb.key
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, name=config.wandb.id,
               sync_tensorboard=True, dir=logdir)
    wandb.config.update(config)

  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'

  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  train_replay = common.Replay(logdir / 'train_episodes', **config.replay, load_directory= loaddir / 'train_episodes')
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
    capacity=config.replay.capacity // 10,
    minlen=config.dataset.length,
    maxlen=config.dataset.length,
  ))
  step = common.Counter(0)
  outputs = [
    common.TerminalOutput(),
    common.JSONLOutput(logdir),
    common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  should_train = common.Every(config.train_every)
  should_log = common.Every(config.log_every)
  video_every = config.video_every
  if video_every <= 0:
    video_every = config.eval_every
  should_video_train = common.Every(video_every)
  should_video_eval = common.Every(config.eval_every)
  should_expl = common.Until(config.expl_until // config.action_repeat)

  agnt = None

  def per_episode(ep, mode):

    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
      for key in config.log_keys_video:
        logger.video(f'{mode}_policy_{key}', ep[key])
        if agnt is not None:
          report = {}
          report[f'full_{mode}_policy_{key}'] = agnt.wm.video_pred(
            agnt.wm.preprocess(ep), key,
            viz_context=config.c_rssm,
            num_eps=1,
            openl_lim=99,
            t_lim=100,
          )
          if agnt.wm.hierarchical:
            report[f'thick_{mode}_policy_{key}'] = agnt.wm.thick_video_pred(
              agnt.wm.preprocess(ep), key,
              viz_context=config.c_rssm,
              num_eps=1, t_lim=100,
              hl_plan=agnt.hl_planner,
            )

            if config.hl_plan and mode == 'eval':
              report[f'hl_first_goal_{mode}'] = agnt.hl_planner.viz_first_goal(agnt)
              report[f'hl_goal_seq_{mode}'] = agnt.hl_planner.viz_last_goal_seq(agnt)

          logger.add(report)
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()
    if agnt is not None:
      agnt.planner.reset()
      if agnt.hl_planner is not None:
        agnt.hl_planner.reset()

  if config.expl_until <= 0:
    config = config.update({"expl_behavior": "greedy"}) # Fewer parameters if no exploration is used

  num_eval_envs = min(config.envs, config.eval_eps)
  train_envs = [make_env('train', config=config) for _ in range(config.envs)]
  eval_envs = [make_env('eval', config=config) for _ in range(num_eval_envs)]
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  prefill = max(0, config.prefill - train_replay.stats['total_steps'])
  assert not prefill, "We assume a replay buffer with sufficient data is provided"
  random_agent = common.RandomAgent(act_space)
  eval_driver(random_agent, episodes=int(config.eval_eps * 10)) # start with more than one to avoid empty set
  eval_driver.reset()

  train_dataset = iter(train_replay.dataset(**config.dataset))
  report_dataset = iter(train_replay.dataset(**config.dataset))
  eval_dataset = iter(eval_replay.dataset(**config.dataset))
  agnt = agent.Agent(config, obs_space, act_space, step)

  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(train_dataset))
  if (logdir / 'variables.pkl').exists():
    print('Loading agent. ')
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      train_agent(next(train_dataset))
  if config.train_plan:
    train_policy = lambda *args: agnt.plan(*args, mode='train')
  else:
    train_policy = lambda *args: agnt.policy(*args, mode='explore' if should_expl(step) else 'train')
  if config.eval_plan:
    eval_policy = lambda *args: agnt.plan(*args, mode='eval')
  else:
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

  def train_step(tran, worker):
    if should_train(step):
      for _ in range(config.train_steps):
        mets = train_agent(next(train_dataset))
        [metrics[key].append(value) for key, value in mets.items()]
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset)), prefix='train')
      logger.write(fps=True)
  train_driver.on_step(train_step)

  while step < config.steps:
    logger.write()
    logger.add(agnt.report(next(eval_dataset)), prefix='eval')
    eval_driver(eval_policy, episodes=config.eval_eps)
    train_driver(train_policy, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')

  logger.write(_print=True)

  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass


if __name__ == '__main__':
  main(sys.argv)
