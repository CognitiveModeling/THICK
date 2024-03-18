import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd

import common
import expl
import math

import numpy as np

from cem_agent import CEMAgent
from common.thick_utils import gen_thick_data, THICKNet, flatten_logits, unflatten_logits, gen_thick_inps, gen_thick_segment_data
from common.other import long_term_return
from hl_mcts import HierarchicalPlanner, DummyPlanner


class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)

    self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    self.wm = WorldModel(config, obs_space, self.act_space, self.tfstep, actor=self._task_behavior.actor)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())
    self._max_t = config['dataset']['length']

    discrete_actions = False
    if hasattr(self.act_space, 'n'):
      discrete_actions = True
      act_dim = self.act_space.n
      # planning with discrete action spaces not implemented, TODO add MCTS here
      self.planner = DummyPlanner(config)
    else:
      act_dim = self.act_space.shape[0]
      self.planner = CEMAgent(config, action_dim=act_dim)
    self.hl_planner = None
    if self.config.hl_plan:
      self.hl_planner = HierarchicalPlanner(config=config, hl_action_dim=config.thick_hl_act_dim,
                                            ll_action_dim=act_dim, discrete_action_space=discrete_actions)

  def plan(self, obs, state=None, mode='train'):
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
      int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action

    latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    rnn_sample = (mode == 'train') or not self.config.eval_ctxt_mean
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], sample, rnn_sample=rnn_sample)
    if self.config.hl_plan:
      action, _, _ = self.hl_planner.plan(hl_model=self.wm, ll_model=self.wm.rssm,
                                          ll_reward_model=self.wm.heads['reward'], model_state=latent,
                                          goal=self.hl_planner.goal)
    else:
      action = self.planner.plan(model=self.wm.rssm, reward_model=self.wm.heads['reward'], model_state=latent)
    if mode == 'eval':
      noise = self.config.eval_noise
    elif mode == 'train':
      noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    rnn_sample = (mode == 'train') or not self.config.eval_ctxt_mean
    latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], sample, rnn_sample=rnn_sample)
    feat = self.wm.rssm.get_feat(latent)
    seq = {'feat': feat, 'ctxt_out':  self.wm.rssm.get_coarse_out(latent) if self.config.c_rssm else feat}
    if mode == 'eval':
      actor = self._task_behavior.actor(seq[self.config.actor_seq_inp])
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(seq[self.config.actor_seq_inp])
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      actor = self._task_behavior.actor(seq[self.config.actor_seq_inp])
      action = actor.sample()
      noise = self.config.expl_noise
    else:
      raise NotImplementedError(mode)
    action = common.action_noise(action, noise, self.act_space)
    context = None
    if 'context' in latent.keys():
      context = latent['context']
    outputs = {'action': action, 'goal_stoch': None, 'context': context}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train(self, data, state=None):
    metrics = {}
    out_state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)

    if self.wm.hierarchical:
      thick_mets = self.wm.train_thick(data, state)
      metrics.update(thick_mets)

    start = outputs['post']
    if not (self.config.train_plan and self.config.eval_plan):
      if self.config.expl_behavior == 'greedy' or not self.config.eval_plan:
        # train task behavior only if we use it
        reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
        metrics.update(self._task_behavior.train(self.wm, start, data['is_terminal'], reward))
      if self.config.expl_behavior != 'greedy':
        mets = self._expl_behavior.train(start, outputs, data)[-1]
        metrics.update({'expl_' + key: value for key, value in mets.items()})

    if self.config.eval_plan or self.config.train_plan:
      if self.hl_planner is not None:
        metrics.update(self.hl_planner.get_mets())
      else:
        metrics.update(self.planner.get_mets())
    return out_state, metrics


  @tf.function
  def report(self, data):
    report = {}
    data = self.wm.preprocess(data)
    for key in self.wm.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      openl_tlim = self._max_t
      openl_min = 5
      if 'scripted' in self.config.task:
        openl_tlim = 100
        openl_min = 3
      report[f'openl_{name}'] = self.wm.video_pred(data, key, viz_context=self.config.c_rssm,
                                                   openl_lim=openl_min, t_lim=openl_tlim)
      if self.wm.hierarchical:
        t_lim = self._max_t
        report[f'thick_{name}'] = self.wm.thick_video_pred(data, key, t_lim=t_lim,
                                                           viz_context=self.config.c_rssm)
    return report

  def get_behavior(self, mode='train'):
    if mode == 'train' or mode == 'eval':
      return self._task_behavior
    elif mode == 'expl':
      return self._expl_behavior
    else:
      raise NotImplementedError(mode)


class WorldModel(common.Module):

  def __init__(self, config, obs_space, act_space, tfstep, actor):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.config = config
    self.tfstep = tfstep
    if self.config.c_rssm:
      self.rssm = common.ContextRSSM(**config.rssm, **config.context, act_dim=act_space.shape[0])
    else:
      self.rssm = common.EnsembleRSSM(**config.rssm)
    self.encoder = common.Encoder(shapes, **config.encoder)

    self.actor = actor

    # fine prediction heads
    self.heads = {}
    self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    self._loss_scales = config.loss_scales
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name

    if config.ctxt_grad_heads == 'none':
      self.ctxt_grad_heads = []
    else:
      self.ctxt_grad_heads = config.ctxt_grad_heads

    # coarse prediction heads
    self.ctxt_heads = {}
    if self.config.c_rssm:
      self.ctxt_heads['decoder'] = common.Decoder(shapes, **config.decoder, name='context')
      self.ctxt_heads['reward'] = common.MLP([], **config.reward_head, name='context')
      self._loss_scales = config.loss_scales
      if config.pred_discount:
        self.ctxt_heads['discount'] = common.MLP([], **config.discount_head, name='context')

    self.model_opt = common.Optimizer('model', **config.model_opt)
    self.hierarchical = config.hierarchical
    discrete = config.rssm.discrete
    self.discrete = discrete
    self.cont_act = False
    if self.hierarchical:
      cont_act = not hasattr(act_space, 'n')
      self.thick_net = THICKNet(
        stoch=config.rssm.stoch,
        hidden=config.thick_feat,
        discrete=discrete,
        action_dim=act_space.shape[0],
        cont_action=cont_act,
        config=config,
      )
      self.cont_act = cont_act
      self.thick_opt = common.Optimizer('thick_net', **config.thick_opt)

  def train(self, data, state=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, *self.heads.values(), *self.ctxt_heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None):
    st_clamp = lambda x: tf.stop_gradient(tf.clip_by_value(x, 0, 1)) + (x - tf.stop_gradient(x))
    data = self.preprocess(data)
    embed = self.encoder(data)
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)

    if self.config.c_rssm:
      ctxt_kl_loss, ctxt_kl_value = self.rssm.kl_loss(post, prior, **self.config.ctxt_kl, prefix='ctxt_')
    else:
      ctxt_kl_loss = 0.0
      ctxt_kl_value = kl_value

    # update gate regularization
    ctxt_sparsity = gate_activity = post['gates'].mean()
    _, max_t, gate_dim = post['gates'].shape
    gate_time_steps = st_clamp(tf.math.reduce_sum(post['gates'], axis=-1))
    likes = {}
    losses = {'kl': kl_loss, 'gate_activity': gate_activity, 'ctxt_sparsity': ctxt_sparsity, 'ctxt_kl': ctxt_kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = tf.cast(dist.log_prob(data[key]), tf.float32)
        likes[key] = like
        losses[key] = -like.mean()
    losses['ctxt_sparsity'] = tf.cast(losses['ctxt_sparsity'], tf.float32)

    c_likes = {}
    for c_name, c_head in self.ctxt_heads.items():
      c_grad_head = (c_name in self.config.ctxt_grad_heads)
      c_inp = self.rssm.get_coarse_out(post)
      c_inp = c_inp if c_grad_head else tf.stop_gradient(c_inp)
      c_out = c_head(c_inp)
      c_dists = c_out if isinstance(c_out, dict) else {c_name: c_out}
      for c_key, c_dist in c_dists.items():
        c_like = tf.cast(c_dist.log_prob(data[c_key]), tf.float32)
        c_likes[f'ctxt_{c_key}'] = c_like
        losses[f'ctxt_{c_key}'] = -c_like.mean()

    model_loss = sum(self._loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value, ctxt_kl=ctxt_kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    for name, value in losses.items():
      scale_i = self._loss_scales.get(name, 1.0)
      metrics[f'{name}_scaled_loss'] = value * scale_i
    metrics['model_kl'] = kl_value.mean()
    metrics['model_ctxt_kl'] = ctxt_kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    metrics['gate_activity'] = gate_activity
    metrics['gate_time_steps'] = gate_time_steps.mean() * max_t
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def preprocess_thick_data(self, data, embed, post, dtype, latent_key, thick_segment_crit, t_lim=-1, num_eps=-1):
    stop_g = lambda x: tf.stop_gradient(x)
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    tfcast = lambda x: tf.cast(x, dtype)

    if t_lim <= 0 or embed.shape[1] < t_lim:
      t_lim = embed.shape[1]
    if num_eps <= 0 or embed.shape[0] < num_eps:
      num_eps = embed.shape[0]

    bin_gates = tf.cast(post['gates'], dtype)
    if self.discrete:
      flat_logits = flatten_logits(post['logit'])
      flat_stoch = flatten_logits(post['stoch'])
    else:
      flat_logits = tfcast(tf.concat([post['mean'], post['std']], -1))
      flat_stoch = tfcast(post['stoch'])
    thick_actions = tfcast(data['action'])[:num_eps, :t_lim]
    thick_latents = tfcast(post[latent_key])
    thick_rewards = tf.expand_dims(tfcast(data['reward'])[:num_eps, :t_lim], -1)
    thick_xs = gen_thick_inps(latents=tfcast(post[latent_key][:num_eps, :t_lim]),
                              stoch=tfcast(post['stoch'])[:num_eps, :t_lim], discrete=self.discrete)
    thick_segments = gen_thick_segment_data(gates=bin_gates[:num_eps, :t_lim],
                                            is_terminal=data['is_terminal'][:num_eps, :t_lim],
                                            thick_segment_crit=thick_segment_crit)
    seq_inputs = {**post}

    thick_inps, targets, loss_masks, seq_targets = gen_thick_data(
      inp_states=stop_g(swap(thick_xs)),
      tar_states=stop_g(swap(flat_logits[:num_eps, :t_lim])),
      tar_actions=swap(thick_actions),
      tar_latent=stop_g(swap(thick_latents)),
      tar_rewards=stop_g(swap(thick_rewards)),
      tar_stoch=stop_g(swap(flat_stoch[:num_eps, :t_lim])),
      gs=stop_g(swap(tf.identity(thick_segments))),
      discount=self.config.discount,
    )
    thick_data = {
      'thick_xs': thick_xs,
      'thick_segments': thick_segments,
      'thick_actions': thick_actions,
    }
    if self.discrete:
      targets['thick_logit'] = unflatten_logits(targets['thick_tar_states'])
    else:
      _, TD = targets['thick_tar_states'].shape
      targets['thick_mean'] = targets['thick_tar_states'][:, :self.config.rssm.stoch]
      targets['thick_std'] = targets['thick_tar_states'][:, self.config.rssm.stoch:]
    return thick_inps, targets, loss_masks, thick_data, seq_inputs, seq_targets

  def thick_loss(self, data, c_rssm, state=None):
    data = self.preprocess(data)
    embed = self.encoder(data)
    dtype = embed.dtype
    post, prior = self.rssm.observe(embed, data['action'], data['is_first'], state)
    if c_rssm:
      latent_key = 'context'
    else:
      latent_key = 'deter'
    thick_inps, targets, loss_masks, _, _, _ = self.preprocess_thick_data(
      data=data, embed=embed, post=post, dtype=dtype, latent_key=latent_key,
      thick_segment_crit=self.config.thick_segment_crit,
    )
    thick_losses, thick_mets = self.thick_net.thick_loss(
      inps=thick_inps,
      targets=targets,
      loss_masks=loss_masks,
    )
    thick_model_loss = sum(self.config.thick_loss_scales.get(k, 1.0) * v for k, v in thick_losses.items())
    thick_metrics = {f'{name}_loss': value for name, value in thick_losses.items()}
    for name, value in thick_mets.items():
      thick_metrics[name] = value
    for name, value in thick_losses.items():
      thick_metrics[f'{name}_scaled_loss'] = value * self.config.thick_loss_scales.get(name, 1.0)
    return thick_model_loss, thick_metrics

  def train_thick(self, data, state=None):
    with tf.GradientTape() as thick_model_tape:
      thick_model_loss, thick_metrics = self.thick_loss(data, c_rssm=self.config.c_rssm, state=state)
    thick_modules = [self.thick_net]
    thick_metrics.update(self.thick_opt(thick_model_tape, thick_model_loss, thick_modules))
    return thick_metrics

  def hl_action_prior(self, start):
    assert len(start['context'].shape)==2 and start['context'].shape[0] == 1, "Only one episode"
    inputs = gen_thick_inps(latents=start['context'], stoch=start['stoch'], discrete=self.discrete)
    hl_act_prior = self.thick_net.get_hl_act_prior(inputs)
    return tf.squeeze(hl_act_prior, 0)

  def hierarchical_img_step(self, start, hl_act=None, num_eps=1, sample_stoch=True):
    _dtype = start['context'].dtype
    _cast = lambda x: tf.cast(x, _dtype)
    thick_max_len = self.config.replay.maxlen
    inputs = gen_thick_inps(latents=start['context'][:num_eps], stoch=start['stoch'][:num_eps], discrete=self.discrete)

    # one high-level step
    if hl_act is None:
      thick_outs = self.thick_net.sample_step(inps=inputs, sample=sample_stoch)
    else:
      thick_outs = self.thick_net.hl_act_step(inps=inputs, hl_act=hl_act, sample=sample_stoch)

    # one low level step
    next_prior_state = {'stoch': thick_outs['thick_stoch'],  'context':  start['context'],
                        'deter': tf.zeros(start['context'].shape[:-1] + (self.config.rssm.deter,), _dtype),  # adding dummy deter
                        'action': thick_outs['thick_act'],
                        }
    ll_out = self.rssm.img_step(prev_state=next_prior_state, prev_action=thick_outs['thick_act'])

    # fill and rename output dict
    next_prior_state['prev_stoch'] = ll_out['prev_stoch']
    next_prior_state['prev_action'] = ll_out['prev_action']
    next_state = {'stoch': ll_out['ctxt_stoch'], 'context':  _cast(ll_out['context']),
                  'pre_ll_stoch': next_prior_state['stoch'], 'pre_ll_context':  _cast(next_prior_state['context']),
                  'pre_ll_ctxt_out': _cast(self.rssm.get_coarse_pred(next_prior_state)),
                  'delta_time': tf.clip_by_value(tf.squeeze(thick_outs['thick_time'], -1), 0, thick_max_len)+1,
                  'action': thick_outs['thick_act'], 'prev_stoch': ll_out['prev_stoch'],
                  'prev_action': ll_out['prev_action'],
                  }
    if self.discrete:
      next_state['logit'] = ll_out['logit']
      next_state['pre_ll_logit'] = thick_outs['thick_logit']
    ctxt_out = self.rssm.get_coarse_pred(next_state)
    reward = self.ctxt_heads['reward'](ctxt_out).mode()
    inter_reward = tf.squeeze(thick_outs['reward'], -1)
    final_state = {**next_state, 'ctxt_out': ctxt_out, 'mean_reward': reward, 'inter_reward': inter_reward}
    if 'discount' in self.ctxt_heads:
      disc_dist = self.ctxt_heads['discount'](final_state['ctxt_out'])
      disc = disc_dist.mean()
      ends = tf.ones_like(disc) - tf.cast(disc_dist.sample(), disc.dtype)
    else:
      disc = self.config.discount * tf.ones(final_state['ctxt_out'].shape[:-1])
      ends = tf.zeros_like(disc)
    final_state['is_terminal'] = ends
    final_state['discount'] = tf.pow(disc, final_state['delta_time'])
    return final_state

  def thick_imagine(self, inp_seq, num_eps=-1):
    hor, batch, _ = inp_seq['context'].shape
    _dtype = inp_seq['context'].dtype
    _cast = lambda x: tf.cast(x, _dtype)
    if num_eps < 0:
      num_eps = batch  # full batch size by default
    thick_max_len = self.config.replay.maxlen
    seq_entries = ['stoch', 'context', 'ctxt_out', 'action', 'mean_reward', 'delta_time', 'inter_reward',
                   'ctxt_stoch', 'prev_stoch', 'prev_action']
    seq = {k: [] for k in seq_entries}
    for t in range(hor):
      next_inp_state = {'stoch': inp_seq['stoch'][t], 'context':  inp_seq['context'][t],
                        'mean_reward': inp_seq['mean_reward'][t], 'action': inp_seq['action'][t],
                        'deter': inp_seq['deter'][t], 'ctxt_stoch': inp_seq['ctxt_stoch'][t],
                        'prev_stoch': inp_seq['prev_stoch'][t], 'prev_action': inp_seq['prev_action'][t]}
      next_inp_state['ctxt_out'] = self.rssm.get_coarse_pred(next_inp_state)
      inputs = gen_thick_inps(latents=next_inp_state['context'][:num_eps], stoch=next_inp_state['stoch'][:num_eps],
                              discrete=self.discrete)
      thick_outs = self.thick_net.sample_step(inps=inputs)
      next_prior_state = {'stoch': thick_outs['thick_stoch'],  'context':  next_inp_state['context'],
                          'deter': next_inp_state['deter'],  # adding dummy deter
                          }
      ll_out = self.rssm.img_step(prev_state=next_prior_state, prev_action=thick_outs['thick_act'])
      full_state = {'stoch': ll_out['ctxt_stoch'], 'context':  _cast(ll_out['context']),
                    'delta_time': tf.clip_by_value(tf.squeeze(thick_outs['thick_time'], -1), 0, thick_max_len)+1,
                    'action': thick_outs['thick_act'], 'ctxt_stoch': ll_out['ctxt_stoch'],
                    'prev_stoch': ll_out['prev_stoch'], 'prev_action': ll_out['prev_action'],
                    }
      ctxt_out = self.rssm.get_coarse_pred(full_state)
      reward = self.ctxt_heads['reward'](ctxt_out).mode()
      inter_reward = tf.squeeze(thick_outs['reward'], -1)
      for key, value in {**full_state, 'ctxt_out': ctxt_out, 'mean_reward': reward, 'inter_reward': inter_reward}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.ctxt_heads:
      disc_dist = self.ctxt_heads['discount'](seq['ctxt_out'])
      disc = disc_dist.mean()
      ends = tf.ones_like(disc) - tf.cast(self.ctxt_heads['discount'](seq['ctxt_out']).sample(), disc.dtype)
    else:
      disc = self.config.discount * tf.ones(seq['ctxt_out'].shape[:-1])
      ends = tf.zeros_like(disc)
    seq['is_terminal'] = ends
    seq['inter_discount'] = tf.pow(disc, seq['delta_time'])
    seq['discount'] = disc
    return seq

  def imagine(self, policy, start, is_terminal, horizon):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    if self.config.c_rssm:
      start['ctxt_out'] = self.rssm.get_coarse_pred(start)
    else:
      start['ctxt_out'] = start['feat']
    start['action'] = tf.zeros_like(policy(start[self.config.actor_seq_inp]).mode())
    seq = {k: [v] for k, v in start.items()}
    for t in range(horizon):
      action = policy(tf.stop_gradient(seq[self.config.actor_seq_inp][-1])).sample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      if self.config.c_rssm:
        ctxt_out = self.rssm.get_coarse_pred(state)
      else:
        ctxt_out = feat
      for key, value in {**state, 'action': action, 'feat': feat, 'ctxt_out': ctxt_out}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      ends = tf.ones_like(disc) - tf.cast(self.heads['discount'](seq['feat']).sample(), disc.dtype)
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first_disc = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first_disc *= self.config.discount
        true_first_end = flatten(is_terminal).astype(ends.dtype)
        disc = tf.concat([true_first_disc[None], disc[1:]], 0)
        ends = tf.concat([true_first_end[None], ends[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
      ends = tf.zeros(seq['feat'].shape[:-1])
    seq['mean_reward'] = self.heads['reward'](seq['feat']).mode()
    seq['inter_reward'] = tf.zeros_like(seq['mean_reward'])  # to match dict of hierarchical imagine
    seq['discount'] = disc
    seq['is_terminal'] = ends
    seq['delta_time'] = tf.ones_like(disc)
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data, key, openl_lim=5, viz_context=False, num_eps=6, t_lim=-1):
    # Video logging of low-level predictions
    # Gifs have the following structure:
    # [          input image          ]
    # [fine prediction reconstruction ]
    # [  mismatch of fine prediction  ]
    # [        predicted reward       ] (green)
    # [      predicted discount       ] (white)
    # [  coarse prediction reconstr.  ]
    # [ mismatch of coarse prediction ]
    # [        context vector         ] (grayscale)
    # [   change in context vector    ] (grayscale)
    # [ inputs with change in context ]

    decoder = self.heads['decoder']
    embed = self.encoder(data)
    dtype = embed.dtype

    if len(embed.shape) == 2: # Only one sequence is passed
      assert num_eps ==1
      embed = tf.expand_dims(embed, 0)
      actions = tf.expand_dims(data['action'], 0)
      is_first = tf.expand_dims(data['is_first'], 0)
      video_truth = tf.expand_dims(data[key], 0)
    else:
      actions = data['action']
      is_first = data['is_first']
      video_truth = data[key]

    episode_len = embed.shape[1]
    if episode_len < openl_lim+1:
      openl_lim = episode_len-1
    if episode_len < t_lim:
      t_lim = episode_len

    truth = video_truth[:num_eps, :t_lim] + 0.5
    states, _ = self.rssm.observe(
        embed[:num_eps, :openl_lim], actions[:num_eps, :openl_lim], is_first[:num_eps, :openl_lim])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:num_eps]
    init = {k: v[:, -1] for k, v in states.items()}

    prior = self.rssm.imagine(actions[:num_eps, openl_lim:t_lim], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode()

    _, _, _, _, C = openl.shape
    grayscale = False
    if C == 1:  # grayscale images
      grayscale = True
    model = tf.concat([recon[:, :openl_lim] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    if grayscale:
      video = tf.concat([self._grayscale_to_RGB(truth), self._grayscale_to_RGB(model), self._grayscale_to_RGB(error)], 2)
    else:
      video = tf.concat([truth, model, error], 2)
    B, T, _, W, C = video.shape

    # append discount and reward viz if there is a head for that
    for head_name, head_c in zip(["reward", "discount"], ["G", "RGB"]):
      if head_name in self.heads:
        val_cl = self.heads[head_name](self.rssm.get_feat(states)).mode()[:num_eps]
        val_ol = self.heads[head_name](self.rssm.get_feat(prior)).mode()
        val = tf.clip_by_value(tf.cast(tf.concat([val_cl[:, :openl_lim], val_ol], 1), dtype), -1, 1)
        video = tf.concat([video, self._viz_value(val, pixel_width=W, color_shade=head_c)], 2)

    # append deterministic hidden states or contexts to video
    if viz_context:
      ctxt_decoder = self.ctxt_heads['decoder']
      ctxt_recon = ctxt_decoder(self.rssm.get_coarse_out(states))[key].mode()[:num_eps]
      ctxt_openl = ctxt_decoder(self.rssm.get_coarse_pred(prior))[key].mode()
      ctxt_model = tf.concat([ctxt_recon[:, :openl_lim] + 0.5, ctxt_openl + 0.5], 1)
      ctxt_error = (ctxt_model - truth + 1) / 2
      if grayscale:
        video = tf.concat([video, self._grayscale_to_RGB(ctxt_model), self._grayscale_to_RGB(ctxt_error)], 2)
      else:
        video = tf.concat([video, ctxt_model, ctxt_error], 2)

      latent_key = 'context'
      latent_dim = self.rssm.ctxt_dim()
    else:
      latent_key = 'deter'
      latent_dim = self.rssm.deter_dim() # only look at deter dim: GateL0RDV2 concats short term dim as well

    latent = tf.concat([states[latent_key][:, :openl_lim, :latent_dim], prior[latent_key][:, :, :latent_dim]], 1)

    BD, TD, DD = latent.shape
    HW_latent = int(math.sqrt(DD))
    if HW_latent == math.sqrt(DD):  # right now only plot squares
      img_latent = self._viz_latent(latent, vid_W=W, color_shade='RGB')
      video = tf.concat([video, img_latent], 2)

      # visualize change in hidden states
      delta_latent = tf.concat(
        [tf.zeros_like(latent[:, 0:1, :]), (latent[:, 1:TD, :] - latent[:, 0:(TD-1), :])],
        1,
      )
      img_delta_latent = self._viz_latent(delta_latent, vid_W=W, color_shade='RGB')
      video = tf.concat([video, img_delta_latent], 2)

      # plot frames with gate opening
      delta_latent = tf.concat(
        [tf.ones_like(latent[:, 0:1, :]), (latent[:, 1:TD, :] - latent[:, 0:(TD - 1), :])],
        1,
      )
      max_delta_latent = tf.clip_by_value(
        tf.math.ceil(
          tf.math.reduce_sum(tf.math.abs(delta_latent),
                             axis=-1,
                             keepdims=True)
        ),
        0,
        1
      )
      img_like_max_delta_latent = tf.repeat(
        tf.repeat(
          tf.repeat(
            tf.expand_dims(tf.expand_dims(max_delta_latent, -1), -1),
            repeats=W,
            axis=2
          ),
          repeats=W,
          axis=3),
        repeats=3,
        axis=-1
      )
      gate_open_truth = truth * img_like_max_delta_latent
      video = tf.concat([video, gate_open_truth], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

  def thick_video_pred(self, data, key, t_lim=-1, viz_context=False, num_eps=6, hl_plan=None, viz_prior=False):
    # Video logging of high- and low-level predictions
    # If hl_plan, then the high-level is used to plan here
    # (for visualizing/debugging THICK PlaNet)
    #
    # Gifs have the following structure:
    # [          input image          ]
    # [         true action           ] (grayscale)
    # [ high-level stoch prediction   ]
    # [  high-level action prediction ]
    # ([high-level goal b4 low-level ]) to visualize THICK PlaNet, if hl_plan
    # ([hl goal after  low-level step]) , if hl_plan
    # ([planned high-level action A_t]) (green), if hl_plan
    # [      post. high-level pred.   ]
    # [  high-level action A_t prior  ] (grayscale)
    # [  high-level action A_t post.  ] (red)
    # [high-level stoch pred. for A_1 ]
    # [high-level action pred. for A_1] (grayscale)
    # ...
    # [high-level stoch pred. for A_N ]
    # [high-level action pred. for A_N] (grayscale)

    # useful definitions
    stop_g = lambda x: tf.stop_gradient(x)
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))

    decoder = self.heads['decoder']
    if viz_context:
      decoder = self.ctxt_heads['decoder']

    # embed video data
    embed = self.encoder(data)
    dtype = embed.dtype

    new_data = {}  # create own data based on preprocessing
    if len(embed.shape) == 2:  # Only one sequence is passed, per episode log
      assert num_eps == 1
      embed = tf.expand_dims(embed, 0)
      new_data['action'] = tf.expand_dims(data['action'], 0)
      new_data['is_first'] = tf.expand_dims(data['is_first'], 0)
      new_data['is_terminal'] = tf.expand_dims(data['is_terminal'], 0)
      new_data['reward'] = tf.expand_dims(data['reward'], 0)
      video_truth = tf.expand_dims(data[key], 0)
    else: # agent log
      new_data['action'] = data['action'][:num_eps]
      new_data['is_terminal'] = data['is_terminal'][:num_eps]
      new_data['is_first'] = data['is_first'][:num_eps]
      new_data['reward'] = data['reward'][:num_eps]
      video_truth = data[key]

    if t_lim <= 0 or embed.shape[1] < t_lim:
      t_lim = embed.shape[1]

    # get targets and gates from GateL0RD
    post, prior = self.rssm.observe(
      embed[:num_eps, :t_lim],
      new_data['action'][:num_eps, :t_lim],
      new_data['is_first'][:num_eps, :t_lim],
    )
    # hidden states
    if viz_context:
      latent_key = 'context'
    else:
      latent_key = 'deter'

    # create THICK input-target pairs
    thick_inps, targets, loss_masks, thick_data, seq_inputs, seq_targets = self.preprocess_thick_data(
      data=new_data, embed=embed,
      post=post, dtype=dtype,
      latent_key=latent_key,
      thick_segment_crit=self.config.thick_segment_crit,
      t_lim=t_lim, num_eps=num_eps,
    )

    if viz_prior:
      _, _, _, _, prior_inputs, _ = self.preprocess_thick_data(
        data=new_data, embed=embed,
        post=prior, dtype=dtype,
        latent_key=latent_key,
        thick_segment_crit=self.config.thick_segment_crit,
        t_lim=t_lim, num_eps=num_eps,
      )
      prior_feats = self.rssm.get_feat(prior_inputs)
      prior_recon_feat = self.heads['decoder'](prior_feats)[key].mode()[:num_eps, :t_lim] + 0.5


    truth = video_truth[:num_eps, :t_lim] + 0.5
    latent = post[latent_key]

    img_dtype = truth.dtype
    t_B, t_S, t_H, W, t_C = truth.shape
    BD, TD, DD = latent.shape

    grayscale = False
    if t_C == 1:  # Grayscale
      grayscale = True

    # THICK data for video truth and save latent state for video reconstruction
    flat_truth = tf.reshape(truth, (num_eps, t_lim, t_H * W * t_C))
    thick_latent, _, _, seq_video_targets = gen_thick_data(
        inp_states=stop_g(swap(latent[:num_eps, :t_lim])),
        tar_states=stop_g(swap(flat_truth)),
        tar_stoch=stop_g(swap(flat_truth)),
        tar_actions=stop_g(swap(latent[:num_eps, :t_lim])),
        tar_latent=stop_g(swap(latent[:num_eps, :t_lim])),
        tar_rewards=stop_g(swap(latent[:num_eps, :t_lim, 0:1])),
        gs=stop_g(swap(tf.identity(thick_data['thick_segments']))),
        discount=self.config.discount,
      )

    # true video at next context change
    truth = truth[:num_eps, :(t_lim - 1)]  # video is 1 time step shorter

    # reconstruction of thick targets using decoder
    seq_thick_tars = seq_targets['state']
    tar_S, tar_B, tar_D = seq_thick_tars.shape
    tar_latent = tf.transpose(tf.reshape(thick_latent, (tar_S, tar_B, DD)), [1, 0, 2])

    # THICK predictions to frames: sampled from logit and appended with corresponding latent
    thick_outs = self.thick_net.sample_step(stop_g(thick_inps))

    # predicted stoch at next context change
    thick_state_outs = thick_outs['thick_stoch']
    stoch_D = thick_state_outs.shape[-1]
    if self.discrete:
      seq_thick_outs = tf.transpose(tf.reshape(thick_state_outs, (tar_S, tar_B, stoch_D, stoch_D)), [1, 0, 2, 3])
    else:
      seq_thick_outs = tf.transpose(tf.reshape(thick_state_outs, (tar_S, tar_B, stoch_D)), [1, 0, 2])

    seq_act_preds = thick_outs['thick_act']
    seq_act_preds = tf.transpose(tf.reshape(seq_act_preds, (tar_S, tar_B, seq_act_preds.shape[-1])), [1, 0, 2])

    pred_outputs = self.thick_net.sample_ll_out(logit=None, mean=None, std=None, stoch_sample=seq_thick_outs,
                                                ctxt_stoch_sample=seq_thick_outs, context=tar_latent,
                                                ll_net=self.rssm, action_pred=seq_act_preds)
    recon_outs = decoder(pred_outputs)[key].mode()[:num_eps, :(t_lim - 1)]
    video_predictions = recon_outs + 0.5

    # predicted action to be performed at next skip
    pixels_for_act_viz = 15
    thick_action_outs = thick_outs['thick_act']
    seq_thick_action = tf.transpose(
      tf.reshape(thick_action_outs, (tar_S, tar_B, thick_action_outs.shape[-1])),
      [1, 0, 2]
    )
    seq_thick_action_hat = seq_thick_action
    if self.cont_act:
      seq_thick_action_hat = seq_thick_action * 0.5 + 0.5 # [-1, 1] -> [0, 1]
    pred_act_video = tf.cast(
      self._viz_actions(seq_thick_action_hat, vid_w=W, block_height=pixels_for_act_viz, color_shade='RGB'),
      img_dtype,
    )
    if grayscale:
      video_predictions = tf.concat([self._grayscale_to_RGB(video_predictions), pred_act_video], 2)
    else:
      video_predictions = tf.concat([video_predictions, pred_act_video], 2)

    if grayscale:
        truth = self._grayscale_to_RGB(truth)

    if viz_prior:
      if grayscale:
        truth = tf.concat([truth, self._grayscale_to_RGB(prior_recon_feat[:num_eps, :(t_lim - 1)])], axis=2)
      else:
        truth = tf.concat([truth, prior_recon_feat[:num_eps, :(t_lim - 1)]], axis=2)

    video_action_truth = self._viz_actions(new_data['action'][:num_eps, :(t_lim - 1)],  vid_w=W, block_height=pixels_for_act_viz, color_shade='RGB')
    video = tf.concat(
      [truth,  # input video
       video_action_truth, # true action
       video_predictions,  # predicted actions
       ],
      axis=2,
    )

    if hl_plan is not None and num_eps == 1:
      # visualize high-level plan
      all_goals = []
      all_hl_acts = []
      goal = None
      for t in range((t_lim-1)):
        last_state = {key: value[0, t:(t+1)] for key, value in post.items()}
        _, goal_hl_act, goal = hl_plan.plan(hl_model=self, ll_model=self.rssm,
                                            ll_reward_model=self.heads['reward'], model_state=last_state, goal=goal)
        all_goals.append(goal)
        all_hl_acts.append(goal_hl_act)
      for keyname in ['pre_ll_ctxt_out', 'ctxt_out']:
        goal_ctxt_out_list = [s[keyname] for s in all_goals]
        goal_ctxt_outs = tf.transpose(tf.stack(goal_ctxt_out_list, 0), [1, 0, 2])
        goal_vid = decoder(goal_ctxt_outs)[key].mode()[0, :(t_lim - 1)] + 0.5
        if grayscale:
          video = tf.concat([video,  self._grayscale_to_RGB(tf.expand_dims(goal_vid, 0))], axis=2)
        else:
          video = tf.concat([video,  tf.expand_dims(goal_vid, 0)], axis=2)
      plan_hl_act = tf.transpose(tf.stack(all_hl_acts, 0), [1, 0, 2])
      plan_hl_act_vid = tf.cast(self._viz_actions(plan_hl_act, vid_w=W, block_height=pixels_for_act_viz, color_shade='G'),
                                img_dtype,)
      video = tf.concat([video,  plan_hl_act_vid], axis=2)

    if self.thick_net.hl_act_dim > 0:
      # high-level prediction based on posterior high-level action A_t
      post_thick_all_out = self.thick_net.posterior_step(stop_g(thick_inps), targets)
      hl_acts_vids = []
      for hl_act_name, hl_color in zip(['post_hl_act_sample', 'prior_hl_act_sample'], ['RGB', 'R']):
        hl_act_sample = post_thick_all_out[hl_act_name]
        seq_hl_act_sample = tf.transpose(
          tf.reshape(hl_act_sample, (tar_S, tar_B, hl_act_sample.shape[-1])), [1, 0, 2],
        )
        seq_hl_act_sample_video = tf.cast(
          self._viz_actions(seq_hl_act_sample, vid_w=W, block_height=pixels_for_act_viz, color_shade=hl_color),
          img_dtype,
        )
        hl_acts_vids.append(seq_hl_act_sample_video)
      all_hl_act_vids = tf.concat(hl_acts_vids, axis=2)

      post_thick_state_outs = post_thick_all_out['thick_stoch']
      post_thick_act = tf.transpose(tf.reshape(post_thick_all_out['thick_act'],  (tar_S, tar_B, post_thick_all_out['thick_act'].shape[-1])), [1, 0, 2])
      if self.discrete:
        post_seq_thick_outs = tf.transpose(tf.reshape(post_thick_state_outs, (tar_S, tar_B, stoch_D, stoch_D)), [1, 0, 2, 3])
      else:
        post_seq_thick_outs = tf.transpose(tf.reshape(post_thick_state_outs, (tar_S, tar_B, stoch_D)), [1, 0, 2])
      post_pred_outputs = self.thick_net.sample_ll_out(
        logit=None, mean=None, std=None, stoch_sample=post_seq_thick_outs, ctxt_stoch_sample=post_seq_thick_outs,
        context=tar_latent, ll_net=self.rssm, action_pred=post_thick_act,
      )
      post_recon_outs = decoder(post_pred_outputs)[key].mode()[:num_eps, :(t_lim - 1)]
      post_video_predictions = post_recon_outs + 0.5
      if grayscale:
        video = tf.concat([video, self._grayscale_to_RGB(post_video_predictions),
                           tf.cast(all_hl_act_vids, video.dtype)], axis=2)
      else:
        video = tf.concat([video, post_video_predictions, tf.cast(all_hl_act_vids, video.dtype)], axis=2)

      # high-level prediction based on all high-level actions A^i
      thick_all_out = self.thick_net.sample_all_step(stop_g(thick_inps))
      for hl_i in range(self.thick_net.hl_act_dim):
        thick_act_outs_i = thick_all_out[f'hl{hl_i}_thick_act']
        seq_thick_act_i = tf.transpose(tf.reshape(thick_act_outs_i, (tar_S, tar_B, thick_act_outs_i.shape[-1])), [1, 0, 2])
        if self.discrete:
          thick_state_outs_i = thick_all_out[f'hl{hl_i}_thick_logit']
          seq_thick_outs_i = tf.transpose(tf.reshape(thick_state_outs_i, (tar_S, tar_B, stoch_D, stoch_D)), [1, 0, 2, 3])
          pred_outputs_i = self.thick_net.sample_ll_out(logit=seq_thick_outs_i, mean=None, std=None, context=tar_latent, ll_net=self.rssm, action_pred=seq_thick_act_i)
        else:
          thick_state_seq_mean = tf.transpose(tf.reshape(thick_all_out[f'hl{hl_i}_thick_mean'], (tar_S, tar_B, stoch_D)), [1, 0, 2])
          thick_state_seq_std = tf.transpose(tf.reshape(thick_all_out[f'hl{hl_i}_thick_std'], (tar_S, tar_B, stoch_D)), [1, 0, 2])
          pred_outputs_i = self.thick_net.sample_ll_out(logit=None, mean=thick_state_seq_mean, std=thick_state_seq_std, context=tar_latent, ll_net=self.rssm, action_pred=seq_thick_act_i)
        recon_outs_i = decoder(pred_outputs_i)[key].mode()[:num_eps, :(t_lim - 1)] + 0.5

        thick_action_outs_i = thick_all_out[f'hl{hl_i}_thick_act']
        seq_thick_action_i = tf.transpose(
          tf.reshape(thick_action_outs_i, (tar_S, tar_B, thick_action_outs_i.shape[-1])),
          [1, 0, 2],
        )
        seq_thick_action_i_hat = seq_thick_action_i
        if self.cont_act:
          seq_thick_action_i_hat = seq_thick_action_i * 0.5 + 0.5
        pred_act_video_i = self._viz_actions(seq_thick_action_i_hat, vid_w=W, block_height=pixels_for_act_viz, color_shade='RGB')
        if grayscale:
          video = tf.concat([video, self._grayscale_to_RGB(recon_outs_i),  tf.cast(pred_act_video_i, video.dtype)], axis=2)
        else:
          video = tf.concat([video, recon_outs_i,  tf.cast(pred_act_video_i, video.dtype)], axis=2)

    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

  # Helper functions for visualization

  @tf.function
  def _grayscale_to_RGB(self, video):
    return tf.repeat(video, 3, -1)

  def _viz_latent(self, latent, vid_W, color_shade='GB'):
    dtype = latent.dtype
    BD, TD, DD = latent.shape
    HW_latent = int(math.sqrt(DD))
    square_latent = latent.reshape((BD, TD, HW_latent, HW_latent))  # latent vector to square matrix
    pixels_latent = int(vid_W / HW_latent)
    square_latent = tf.repeat(square_latent, pixels_latent, 2)  # same height as images
    square_latent = tf.repeat(square_latent, pixels_latent, 3)  # same width as images
    square_latent = tf.expand_dims(square_latent, -1) * 0.5 + 0.5
    img_latent = self._viz_color_shade(square_latent, color_shade=color_shade)

    return self._viz_fill(img_latent, vid_W, vid_W, dtype)  # append zeroes if not 64x64

  def _vid_from_mask(self, mask, mask_H, mask_W, mask_C, imd_dtype):
    for dim in [mask_H, mask_W, mask_C]:
      mask = tf.repeat(tf.expand_dims(mask, -1), repeats=dim, axis=-1)
    return tf.cast(mask, dtype=imd_dtype)

  def _viz_fill(self, img, vid_H, vid_W, dtype):
    img_B, img_T, img_H, img_W, img_C = img.shape
    if img_H < vid_H:
      h_zeros = tf.zeros((img_B, img_T, vid_H - img_H, img_W, img_C), dtype)
      img = tf.concat([img, h_zeros], 2)
    if img_W < vid_W:
      w_zeros = tf.zeros((img_B, img_T, img.shape[2], vid_W - img_W, img_C), dtype)
      img = tf.concat([img, w_zeros], 3)
    return img

  def _viz_actions(self, act, vid_w, color_shade='GB', block_height=-1):
    dtype = act.dtype
    aB, aT, aD = act.shape
    block_width= int(vid_w/aD)
    if block_height <= 0:
      block_height = block_width
    act = tf.expand_dims(act, 2) # B x T x H X W
    act = tf.repeat(act, block_height, 2)
    act = tf.repeat(act, block_width, 3)
    act = tf.expand_dims(act, -1) * 0.5 + 0.5
    img = self._viz_color_shade(act, color_shade=color_shade)
    return self._viz_fill(img, block_height, vid_w, dtype)  # append zeroes

  def _viz_color_shade(self, x, color_shade='GB'):
    # assumes x last index encodes color and is in [0, 1]
    x_R = x
    x_G = x
    x_B = x
    if 'R' not in color_shade:
      x_R = tf.zeros_like(x)
    if 'G' not in color_shade:
      x_G = tf.zeros_like(x)
    if 'B' not in color_shade:
      x_B = tf.zeros_like(x)
    x_RGB = tf.concat([x_R, x_G, x_B], -1)
    return x_RGB

  def _viz_value(self, value, pixel_width, pixel_height=10, color_shade='RG'):
    assert len(value.shape) == 2
    value_H = tf.repeat(tf.expand_dims(value, -1), pixel_height, -1)  # same height as images
    value_HW = tf.repeat(tf.expand_dims(value_H, -1), pixel_width, -1)  # same height as images
    value_HWC = tf.expand_dims(value_HW, -1) * 0.5 + 0.5
    value_img = self._viz_color_shade(value_HWC, color_shade=color_shade)
    return value_img


class ActorCritic(common.Module):

  def __init__(self, config, act_space, tfstep):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})

    self.actor = common.MLP(act_space.shape[0], **self.config.actor)
    self.critic = common.MLP([], **self.config.critic)

    if self.config.thick_dreamer:
      self.coarse_critic = common.MLP([], **self.config.critic)
      if self.config.slow_target:
        self.coarse_target_critic = common.MLP([], **self.config.critic)
      else:
        self.coarse_target_critic = self.coarse_critic

    if self.config.slow_target:
      self._target_critic = common.MLP([], **self.config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic

    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def train(self, world_model, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with tf.GradientTape() as actor_tape:
      seq = world_model.imagine(self.actor, start, is_terminal, hor)
      reward = reward_fn(seq)
      seq['reward'], mets1 = self.rewnorm(reward)
      mets1 = {f'reward_{k}': v for k, v in mets1.items()}
      target, mets2 = self.target(seq)
      actor_loss, mets3 = self.actor_loss(seq, target)
    with tf.GradientTape() as critic_tape:
      if self.config.thick_dreamer:
        hl_seq = world_model.thick_imagine(inp_seq=seq)
        hl_reward = hl_seq['mean_reward']
        hl_seq['reward'], _ = self.rewnorm(hl_reward)
        target, mets2, long_target = self.thick_dreamer_target(seq=seq, hl_seq=hl_seq)
        critic_loss, mets4 = self.thick_dreamer_loss(seq, target)
      else:
        target, mets2 = self.target(seq)
        critic_loss, mets4 = self.critic_loss(seq, target)

    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    critic_modules = [self.critic]
    if self.config.thick_dreamer:
      critic_modules = [self.critic, self.coarse_critic]
    metrics.update(self.critic_opt(critic_tape, critic_loss, critic_modules))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    critic_key = 'feat'
    policy = self.actor(tf.stop_gradient(seq[self.config.actor_seq_inp][:-2]))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq[critic_key][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq[critic_key][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2

    dist_fine = self.critic(seq['feat'][:-1])
    target_fine = tf.stop_gradient(target)
    weight_fine = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist_fine.log_prob(target_fine) * weight_fine[:-1]).mean()
    metrics = {'critic': dist_fine.mode().mean()}

    return critic_loss, metrics

  def thick_dreamer_loss(self, seq, target):
    dist_fine = self.critic(seq['feat'][:-1])
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist_fine.log_prob(target) * weight[:-1]).mean()

    dist_coarse = self.coarse_critic(seq['ctxt_out'][:-1])
    critic_loss_coarse = -(dist_coarse.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist_fine.mode().mean(), 'critic_coarse': dist_coarse.mode().mean()}
    critic_loss = critic_loss + critic_loss_coarse
    return critic_loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat']).mode()
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def thick_dreamer_target(self, seq, hl_seq):
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    critic_key = 'feat'
    value = self._target_critic(seq[critic_key]).mode()

    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)

    hl_reward = tf.cast(hl_seq['reward'], tf.float32)
    hl_interreward = tf.cast(hl_seq['inter_reward'], tf.float32)
    hl_discount = tf.cast(hl_seq['discount'], tf.float32)
    hl_interdiscount = tf.cast(hl_seq['inter_discount'], tf.float32)
    hl_value = self.coarse_target_critic(hl_seq['ctxt_out']).mode()
    hl_target, long_value = long_term_return(hl_reward[:-1], hl_value[:-1], hl_discount[:-1], hl_interreward[:-1], hl_interdiscount[:-1])

    full_target = self.config.critic_psi * target + (1 - self.config.critic_psi) * hl_target

    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_ll_target'] = target.mean()
    metrics['critic_target'] = full_target.mean()
    metrics['critic_hl_slow'] = hl_value.mean()
    metrics['critic_hl_target'] = hl_target.mean()

    return full_target, metrics, long_value

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)

        if self.config.thick_dreamer:
          for s, d in zip(self.coarse_critic.variables, self.coarse_target_critic.variables):
            d.assign(mix * s + (1 - mix) * d)

      self._updates.assign_add(1)
