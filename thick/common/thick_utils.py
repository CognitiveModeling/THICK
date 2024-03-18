import common
import math
import tensorflow as tf

from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd

from common.nets import get_act
from common.other import static_scan


class THICKNet(common.Module):

  def __init__(
          self,
          stoch,
          hidden,
          discrete,
          action_dim,
          cont_action,
          config,
          act='elu',
          int_layers=3,
          min_std=0.1,
  ):
    super().__init__()

    self._stoch = stoch
    self._hidden = hidden
    self._discrete = discrete
    self._action_dim = action_dim
    self._cont_action = cont_action
    self._act = get_act(act)
    self.config = config
    self._int_layers = int_layers
    self.hl_act_dim = config.thick_hl_act_dim
    self._std_act = 'softplus'
    self._min_std = min_std

    if self.hl_act_dim > 0:
      self.hl_act_prior = common.MLP(
        shape=[self.hl_act_dim], layers=3, units=200, act='elu', norm='none', name='hl_act_prior', dist='onehot',
      )
      self.hl_act_post = common.MLP(
        shape=[self.hl_act_dim], layers=3, units=200, act='elu', norm='none', name='hl_act_post', dist='onehot',
      )

    self.thick_heads = {}
    readout_layers = self.config.thick_readouts.layers
    readout_units = self.config.thick_readouts.units
    if cont_action:
      self.thick_heads['action'] = common.MLP(shape=[action_dim], layers=readout_layers, units=readout_units,
                                               act='elu', norm='none', name='thick_action', dist='trunc_normal')
    else:
      self.thick_heads['action'] = common.MLP(shape=[action_dim], layers=readout_layers, units=readout_units,
                                              act='elu', norm='none', name='thick_action', dist='onehot')
    self.thick_heads['time'] = common.MLP(shape=[1], layers=readout_layers, units=readout_units,
                                          act='elu', norm='none', name='thick_state', dist='pos_clamp_normal')
    self.thick_heads['reward'] = common.MLP(shape=[1], layers=readout_layers, units=readout_units, act='elu',
                                            norm='none', name='thick_reward', dist='mse')

    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    if self.config.thick_act_pred_loss == "CCEL_logits":
      self._CCEL = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
    else:
      self._CCEL = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    self._MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

  @tf.function
  def forward_thick_body(self, x):
    for l_i,l in enumerate(range(self._int_layers)):
      x = self.get(f'thick_layer{l_i}', tfkl.Dense, self._hidden)(x)
      x = self._act(x)
    return x

  def _state_pred_layers(self, x):
    if self._discrete:
      x = self.get('thick_logit_pre', tfkl.Dense, self._hidden)(x)
      x = self._act(x)
      x = self.get('thick_logit_out', tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'thick_logit': logit}
    else:
      x = self.get('thick_gauss_pre', tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          f'softplus': lambda: tf.nn.softplus(std),
          f'sigmoid': lambda: tf.nn.sigmoid(std),
          f'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {f'thick_mean': mean, f'thick_std': std}

  def _hl_act_prior_layer(self, x):
    x = self.get('hl_act_prior_pre', tfkl.Dense, self._hidden)(x)
    x = self._act(x)
    logit = self.get('hl_act_prior_out', tfkl.Dense, self.hl_act_dim, None)(x)
    return {'hl_prior_logit': logit}

  def _hl_act_post_layer(self, x):
    x = self.get('hl_act_post_pre', tfkl.Dense, self._hidden)(x)
    x = self._act(x)
    logit = self.get('hl_act_post_out', tfkl.Dense, self.hl_act_dim, None)(x)
    return {'hl_post_logit': logit}

  @tf.function
  def sample_step(self, inps, sample=True):
    inp_dtype = inps.dtype
    _cast = lambda x: tf.cast(x, inp_dtype)
    if self.hl_act_dim > 0:
      hl_prior_logits = self._hl_act_prior_layer(inps)
      prior_dist_hl_act = self.get_HL_act_dist(hl_prior_logits, prob_name='hl_prior')
      full_inps = tf.concat([inps, _cast(prior_dist_hl_act.sample())], -1) # sample from prior
    else:
      full_inps = inps
    thick_feat = self.forward_thick_body(full_inps)
    stats = self._state_pred_layers(x=thick_feat)
    dist = self.get_dist(stats, discrete=self._discrete, prob_name='thick')
    stoch = dist.sample() if sample else dist.mode()
    outs = {'thick_stoch': stoch, **stats}
    if self.config.thick_readouts.sample_skip:
      if self._discrete:
        stoch = flatten_logits(stoch)
      readout_inps = tf.stop_gradient(tf.concat([inps, _cast(stoch)], -1))
    else:
      readout_inps = thick_feat
    outs['thick_act'] = self.thick_heads['action'](readout_inps).sample()
    outs['thick_time'] = self.thick_heads['time'](readout_inps).sample()
    outs['thick_time_mode'] = self.thick_heads['time'](readout_inps).mode()
    outs['reward'] = self.thick_heads['reward'](readout_inps).mode()
    return outs

  def mode_step(self, inps, sample=True):
    inp_dtype = inps.dtype
    _cast = lambda x: tf.cast(x, inp_dtype)
    if self.hl_act_dim > 0:
      hl_prior_logits = self._hl_act_prior_layer(inps)
      prior_dist_hl_act = self.get_HL_act_dist(hl_prior_logits, prob_name='hl_prior')
      full_inps = tf.concat([inps, _cast(prior_dist_hl_act.mode())], -1) # mode of prior
    else:
      full_inps = inps
    thick_feat = self.forward_thick_body(full_inps)
    stats = self._state_pred_layers(x=thick_feat)
    dist = self.get_dist(stats, discrete=self._discrete, prob_name='thick')
    stoch = dist.sample() if sample else dist.mode()
    outs = {'thick_stoch': stoch, **stats}
    if self.config.thick_readouts.sample_skip:
      if self._discrete:
        stoch = flatten_logits(stoch)
      readout_inps = tf.stop_gradient(tf.concat([inps, _cast(stoch)], -1))
    else:
      readout_inps = thick_feat
    outs['thick_act'] = self.thick_heads['action'](readout_inps).sample()
    outs['thick_time'] = self.thick_heads['time'](readout_inps).sample()
    outs['thick_time_mode'] = self.thick_heads['time'](readout_inps).mode()
    outs['reward'] = self.thick_heads['reward'](readout_inps).mode()
    return outs

  def get_hl_act_prior(self, inps):
    assert self.hl_act_dim > 0
    hl_prior_logits = self._hl_act_prior_layer(inps)['hl_prior_logit']
    return tf.nn.softmax(hl_prior_logits, -1)

  def hl_act_step(self, inps, hl_act, sample=True):
    inp_dtype = inps.dtype
    _cast = lambda x: tf.cast(x, inp_dtype)
    full_inps = tf.concat([inps, _cast(hl_act)], -1) # sample from prior
    thick_feat = self.forward_thick_body(full_inps)
    stats = self._state_pred_layers(x=thick_feat)
    dist = self.get_dist(stats, discrete=self._discrete, prob_name='thick')
    stoch = dist.sample() if sample else dist.mode()
    outs = {'thick_stoch': stoch, **stats}
    if self.config.thick_readouts.sample_skip:
      if self._discrete:
        stoch = flatten_logits(stoch)
      readout_inps = tf.stop_gradient(tf.concat([inps, _cast(stoch)], -1))
    else:
      readout_inps = thick_feat
    outs['thick_act'] = self.thick_heads['action'](readout_inps).sample()
    outs['thick_time'] = self.thick_heads['time'](readout_inps).sample()
    outs['reward'] = self.thick_heads['reward'](readout_inps).mode()
    return outs

  def posterior_step(self, inps, targets, sample=True):
    assert self.hl_act_dim > 0
    inp_dtype = inps.dtype
    _cast = lambda x: tf.cast(x, inp_dtype)
    latent_posterior = targets['thick_latent']
    stoch_posterior = targets['thick_stoch']
    inps_posterior = tf.concat([inps, _cast(latent_posterior), _cast(stoch_posterior)], -1)

    # posterior
    hl_post_logits = self._hl_act_post_layer(inps_posterior)
    post_dist_hl_act = self.get_HL_act_dist(hl_post_logits, prob_name='hl_post')
    # prior
    hl_prior_logits = self._hl_act_prior_layer(inps)
    prior_dist_hl_act = self.get_HL_act_dist(hl_prior_logits, prob_name='hl_prior')

    prior_hl_act_sample = prior_dist_hl_act.sample()
    post_hl_act_sample = post_dist_hl_act.sample()

    full_inps = tf.concat([inps, _cast(post_hl_act_sample)], -1) # sample from posterior
    thick_feat = self.forward_thick_body(full_inps)
    stats = self._state_pred_layers(x=thick_feat)
    dist = self.get_dist(stats, discrete=self._discrete, prob_name='thick')
    stoch = dist.sample() if sample else dist.mode()
    outs = {'prior_hl_act_sample': prior_hl_act_sample,
            'post_hl_act_sample': post_hl_act_sample,
            'thick_stoch': stoch, **stats}
    if self.config.thick_readouts.sample_skip:
      if self._discrete:
        stoch = flatten_logits(stoch)
      readout_inps = tf.stop_gradient(tf.concat([inps, _cast(stoch)], -1))
    else:
      readout_inps = thick_feat
    outs['thick_act'] = self.thick_heads['action'](readout_inps).sample()
    outs['thick_time'] = self.thick_heads['time'](readout_inps).sample()
    outs['reward'] = self.thick_heads['reward'](readout_inps).mode()
    return outs

  @tf.function
  def sample_all_step(self, inps, sample=True):
    assert self.hl_act_dim > 0
    B, D = inps.shape
    dtype = inps.dtype
    inp_dtype = inps.dtype
    _cast = lambda x: tf.cast(x, inp_dtype)

    outs = {}
    for hl_i in range(self.hl_act_dim):
      hl_idx = tf.ones(B, dtype=tf.int32) * hl_i
      hl_act = tf.one_hot(hl_idx, self.hl_act_dim, dtype=dtype)
      inps_i = tf.concat([inps, hl_act], -1)
      thick_feat = self.forward_thick_body(inps_i)
      stats = self._state_pred_layers(x=thick_feat)
      dist = self.get_dist(stats, discrete=self._discrete, prob_name='thick')
      stoch = dist.sample() if sample else dist.mode()
      if self._discrete:
        outs[f'hl{hl_i}_thick_logit'] = stats['thick_logit']
      else:
        outs[f'hl{hl_i}_thick_mean'] = stats['thick_mean']
        outs[f'hl{hl_i}_thick_std'] = stats['thick_std']
      outs[f'hl{hl_i}_thick_stoch'] = stoch

      if self.config.thick_readouts.sample_skip:
        if self._discrete:
          stoch = flatten_logits(stoch)
        readout_inps = tf.stop_gradient(tf.concat([inps, _cast(stoch)], -1))
      else:
        readout_inps = thick_feat
      outs[f'hl{hl_i}_thick_act'] = self.thick_heads['action'](readout_inps).sample()
      outs[f'hl{hl_i}_thick_time'] = self.thick_heads['time'](readout_inps).sample()
      outs[f'hl{hl_i}_thick_reward'] = self.thick_heads['reward'](readout_inps).mode()
    return outs

  @tf.function
  def thick_loss(self, inps, targets, loss_masks):
    inp_dtype = inps.dtype
    _cast = lambda x: tf.cast(x, inp_dtype)

    thick_losses = {}
    thick_metrics = {}

    loss_mask = tf.squeeze(tf.cast(loss_masks['time'], tf.float32), -1)

    if self.config.thick_hl_act_dim > 0:
      latent_posterior = targets['thick_latent']
      stoch_posterior = targets['thick_stoch']
      inps_posterior = tf.concat([inps, _cast(latent_posterior), _cast(stoch_posterior)], -1)

      # posterior
      hl_post_logit_stats = self._hl_act_post_layer(inps_posterior)
      post_dist_hl_act = self.get_HL_act_dist(hl_post_logit_stats, prob_name='hl_post')
      # prior
      hl_prior_logit_stats = self._hl_act_prior_layer(inps)
      prior_dist_hl_act = self.get_HL_act_dist(hl_prior_logit_stats, prob_name='hl_prior')

      post_hl_act_logit = hl_post_logit_stats['hl_post_logit']
      prior_hl_act_logit = hl_prior_logit_stats['hl_prior_logit']
      hl_post = {'hl_act_logit': post_hl_act_logit}
      hl_prior = {'hl_act_logit': prior_hl_act_logit}
      hl_action_kl_loss, hl_action_kl_value = self.kl_loss(post=hl_post, prior=hl_prior, loss_mask=loss_mask,
                                                           discrete=True, prob_name='hl_act',
                                                           **self.config.thick_hl_act_kl)
      thick_losses['thick_hl_act_kl'] = hl_action_kl_loss

      post_hl_act_sample = post_dist_hl_act.sample()
      full_inps = tf.concat([inps, _cast(post_hl_act_sample)], -1)

      # log entropies
      thick_metrics['hl_act_prior_ent'] = prior_dist_hl_act.entropy()
      thick_metrics['hl_act_post_ent'] = post_dist_hl_act.entropy()
    else:
      full_inps = inps

    thick_feat = self.forward_thick_body(full_inps)

    stats = self._state_pred_layers(x=thick_feat)
    kl_loss, kl_value = self.kl_loss(targets, stats, loss_mask, discrete=self._discrete, prob_name='thick',
                                     **self.config.thick_kl)
    thick_losses['thick_kl'] = kl_loss

    if self.config.thick_readouts.sample_skip:
      thick_stoch = self.get_dist(stats, discrete=self._discrete, prob_name='thick').sample()
      if self._discrete:
        thick_stoch = flatten_logits(thick_stoch)
      readout_inps = tf.stop_gradient(tf.concat([inps, _cast(thick_stoch)], -1))
    else:
      readout_inps = thick_feat

    target_act = tf.cast(targets['thick_act'], tf.float32)
    act_dist = self.thick_heads['action'](tf.cast(readout_inps, tf.float32))
    if self._cont_action:
      thick_act_loss = -1.0 * (act_dist.log_prob(target_act))
    elif self.config.thick_act_pred_loss == "CCEL_samples":
      thick_act_loss = self._CCEL(target_act, act_dist.sample())
    elif self.config.thick_act_pred_loss == "CCEL_probs":
      thick_act_loss = self._CCEL(target_act, act_dist.get_probs(target_act.shape))
    elif self.config.thick_act_pred_loss == "CCEL_logits":
      thick_act_loss = self._CCEL(target_act, act_dist.get_logits(target_act.shape))
    elif self.config.thick_act_pred_loss == "NLL":
      thick_act_loss = -1.0 * (act_dist.log_prob(target_act))
    elif self.config.thick_act_pred_loss == "MSE_samples":
      thick_act_loss = self._MSE(target_act, act_dist.sample())
    else:
      raise NotImplementedError(self.config.thick_act_pred_loss)
    masked_thick_act_loss = thick_act_loss * loss_mask
    thick_losses['thick_act'] = tf.math.reduce_mean(masked_thick_act_loss)
    thick_metrics['thick_act_entropy'] = act_dist.entropy()

    time_dist = self.thick_heads['time'](tf.cast(readout_inps, tf.float32))
    time_logprobs = (time_dist.log_prob(tf.cast(targets['thick_time'], tf.float32)) * loss_mask)
    thick_losses['thick_time'] = -(time_logprobs).mean()
    thick_metrics['thick_mean_time_pred'] = time_dist.mean()

    reward_dist = self.thick_heads['reward'](tf.cast(readout_inps, tf.float32))
    reward_logprob = (reward_dist.log_prob(tf.cast(targets['thick_inter_returns'], tf.float32)) * loss_mask)
    thick_losses['thick_reward'] = -(reward_logprob.mean())
    thick_metrics['thick_mean_reward_pred'] = reward_dist.mean()

    return thick_losses, thick_metrics

  def sample_ll_out(self, logit, mean, std, context, ll_net, ctxt_stoch_sample=None, stoch_sample=None, action_pred=None):
    if stoch_sample is None and ctxt_stoch_sample is None:
      if self._discrete:
        logit = tf.cast(logit, tf.float32)
        dist = tfd.Independent(common.OneHotDist(logit), 1)
      else:
        mean = tf.cast(mean, tf.float32)
        std = tf.cast(std, tf.float32)
        dist = tfd.MultivariateNormalDiag(mean, std)
      stoch_sample = dist.sample()
      ctxt_stoch_sample = dist.sample()
    ll_state = {'context': context, 'prev_action': action_pred}
    if stoch_sample is not None:
      ll_state['prev_stoch'] = flatten_logits(stoch_sample)
    if ctxt_stoch_sample is not None:
      ll_state['ctxt_stoch'] = ctxt_stoch_sample
    return ll_net.get_coarse_pred(ll_state)

  def get_dist(self, state, discrete, prob_name):
    if discrete:
      logit_name = prob_name + '_logit'
      logit = state[logit_name]
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state[f'{prob_name}_mean'], state[f'{prob_name}_std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  def get_HL_act_dist(self, state, prob_name):
    logit_name = prob_name + '_logit'
    logit = state[logit_name]
    logit = tf.cast(logit, tf.float32)
    dist = common.OneHotDist(logit)
    return dist

  def kl_loss(self, post, prior, loss_mask, forward, balance, free, free_avg, discrete, prob_name):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)

    if prob_name == 'hl_act':
      dist = lambda p: self.get_HL_act_dist(p, prob_name=prob_name)
    else:
      dist = lambda p: self.get_dist(p, discrete=discrete, prob_name=prob_name)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs), dist(rhs))
      value = value * loss_mask
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = kld(dist(lhs), dist(sg(rhs)))
      value_rhs = kld(dist(sg(lhs)), dist(rhs))
      value_lhs = value = value_lhs * loss_mask
      value_rhs = value_rhs* loss_mask
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


def flatten_logits(logits):
  if len(logits.shape) == 4:
    S, B, D1, D2 = logits.shape
    assert D1 == D2
    return tf.reshape(logits, (S, B, int(D1 * D2)))
  else:
    assert len(logits.shape) == 3
    SB, D1, D2 = logits.shape
    assert D1 == D2
    return tf.reshape(logits, (SB, int(D1 * D2)))


def unflatten_logits(logits):
  if len(logits.shape) == 2:
    B, D = logits.shape
    return tf.reshape(logits, (B, int(math.sqrt(D)), int(math.sqrt(D))))
  else:
    assert len(logits.shape) == 3
    S, B, D = logits.shape
    return tf.reshape(logits, (S, B, int(math.sqrt(D)), int(math.sqrt(D))))


def gen_thick_inps(latents, stoch, discrete=True):
  if discrete:
    shape = stoch.shape[:-2] + [stoch.shape[-2] * stoch.shape[-1]]  # flatten stoch
    observ = tf.cast(tf.reshape(stoch, shape), latents.dtype)
  else:
    observ = tf.cast(stoch, latents.dtype)
  return tf.concat([observ, latents], axis=-1)


def gen_thick_segment_data(gates, is_terminal, thick_segment_crit):
  if 'ends' in thick_segment_crit:
    dtype = gates.dtype
    is_terminal = tf.cast(tf.expand_dims(is_terminal, -1), dtype=dtype)
    return tf.concat([gates, is_terminal], axis=-1)
  return gates

def gen_thick_data(xs, ys, gs):
  # xs are inputs
  # ys are potential targets (typically subset of input)
  # gs are gate openings that determine the skips
  x_S, x_B, x_D = xs.shape
  y_S, y_B, y_D = ys.shape
  dtype = ys.dtype
  targets = tf.ones((x_S-1, y_B, y_D), dtype=dtype) * 99
  g_mask = None
  for t in range(x_S-1):
    # Predict targets y at time t-1 for gate openings at time t.
    targets, g_mask = thick_data_gen_step(ys[t, :, :], gs[t+1, :, :], targets, t, g_mask)

  tars = tf.reshape(targets, ((x_S-1) * y_B, y_D))
  inps = tf.reshape(xs[:(x_S-1), :, :], ((x_S-1) * x_B, x_D))

  # For loss ignore predictions for time steps where no gate openings followed
  # targets_mask = 1.0 - g_mask
  g_mask_reshaped = tf.reshape(g_mask, (int((x_S-1) * y_B), y_D))
  g_mask_reshaped = tf.cast(g_mask_reshaped, tf.float32)
  g_mask_reshaped = tf.math.reduce_sum(g_mask_reshaped, axis=-1, keepdims=True)
  g_mask_reshaped = tf.clip_by_value(g_mask_reshaped, 0, 1)
  loss_mask = 1.0 - g_mask_reshaped
  
  return inps, tars, loss_mask, targets


def thick_data_gen_step(x_t, g_t, targets, t, g_mask=None):
  dtype = x_t.dtype

  # g_mask tracks which past inputs do not have a target assigned yet
  # g_mask at index i == 1 : Input at i still needs a target
  # g_mask at index j == 0 : Input at j already has a target assigned
  t_S, t_B, t_D = targets.shape
  x_B, x_D = x_t.shape
  g_B, _ = g_t.shape

  assert g_mask is not None or t == 0
  assert x_B == t_B and x_D == t_D and x_B == g_B

  # clamped max gate activation in shape of x_t
  g_clamp = tf.clip_by_value(tf.math.reduce_sum(g_t, axis=-1, keepdims=True), 0, 1)
  # Reshape tensor to fit output dim
  g_act = tf.repeat(g_clamp, repeats=x_D, axis=-1)
  # g_act repeated until current time step
  g_seq_t = tf.repeat(tf.expand_dims(g_act, 0), repeats=(t + 1), axis=0)
  # g_mask gets ones appended
  if t == 0:
    g_mask_prime = tf.ones((1, t_B, t_D), dtype=dtype)
  else:
    g_mask_prime = tf.concat([g_mask, tf.ones((1, t_B, t_D), dtype=dtype)], axis=0)
  # if the gate is opened (g_act/g_seq_t) then targets are updated at every index where the new g_mask is 1
  updates = g_seq_t * g_mask_prime
  if t == t_S:
    updates_padded = updates
  else:
    updates_padded = tf.concat([updates, tf.zeros((t_S - (t + 1), t_B, t_D), dtype=dtype)], axis=0)
  x_seq = tf.repeat(tf.expand_dims(x_t, 0), repeats=t_S, axis=0)
  updated_targets = (1 - updates_padded) * targets + updates_padded * x_seq
  # update g_mask, everywhere that a target was assigned the g_mask receives a new 0
  updated_g_mask = (1 - updates) * g_mask_prime
  return updated_targets, updated_g_mask


def gen_thick_data(inp_states, tar_states, tar_actions, tar_latent, tar_rewards, tar_stoch, gs, discount=0.99):
  # inp_states are the input states of the system
  # tar_states are potential targets (typically subset of input)
  # tar_actions are the actions conducted at the target states
  # gs are gate openings that determine the skips
  x_S, x_B, x_D = inp_states.shape
  y_S, y_B, y_D = tar_states.shape
  a_S, a_B, a_D = tar_actions.shape
  l_S, l_B, l_D = tar_latent.shape
  s_S, s_B, s_D = tar_stoch.shape

  dtype = tar_states.dtype
  target_states = tf.ones((x_S-1, y_B, y_D), dtype=dtype) * 99
  target_actions = tf.zeros((x_S-1, a_B, a_D), dtype=dtype)
  target_times = tf.ones((x_S-1, y_B, 1), dtype=dtype) * 99
  target_latent = tf.ones((x_S-1, y_B, l_D), dtype=dtype) * 99
  target_stoch = tf.ones((s_S-1, s_B, s_D), dtype=dtype) * 99

  g_mask_state = None
  g_mask_action = None
  g_mask_time = None
  g_mask_latent = None
  g_mask_stoch = None
  for t in range(x_S-1):
    # Predict target states and actions at time t for gate openings at time t+1.
    target_states, g_mask_state = thick_data_gen_step(tar_states[t, :, :], gs[t+1, :, :], target_states, t, g_mask_state)
    target_actions, g_mask_action = thick_data_gen_step(tar_actions[t+1, :, :], gs[t+1, :, :], target_actions, t, g_mask_action)
    # last target times + 1, starting with zeros
    target_times, g_mask_time = thick_data_gen_step(tf.ones((y_B, 1), dtype=dtype) * t, gs[t+1, :, :], target_times, t, g_mask_time)
    target_latent, g_mask_latent = thick_data_gen_step(tar_latent[t+1, :, :], gs[t+1, :, :], target_latent, t, g_mask_latent)
    target_stoch, g_mask_stoch = thick_data_gen_step(tar_stoch[t, :, :], gs[t+1, :, :], target_stoch, t, g_mask_stoch)

  # targets as sequences (seq x batch x dim)
  seq_targets = {
    'state': target_states,
    'action': target_actions,
    'time': target_times,
    'latent': target_latent,
    'stoch': target_stoch,
  }

  discounts = (1.0-tf.clip_by_value(tf.math.reduce_sum(gs[1:, :, :], axis=-1, keepdims=True), 0, 1))*discount
  inter_returns = static_scan(lambda agg, cur: cur[0] + cur[1] * agg, (tar_rewards[:-1], discounts),
                              tf.zeros_like(discounts[-1]), reverse=True)

  # targets as batches (batch x dim)
  tars_stoch = tf.reshape(target_stoch, ((x_S-1) * y_B, s_D))
  tars_latent = tf.reshape(target_latent, ((x_S-1) * y_B, l_D))
  tars_states = tf.reshape(target_states, ((x_S-1) * y_B, y_D))
  tars_actions = tf.reshape(target_actions, ((x_S-1) * a_B, a_D))
  t_asc = tf.expand_dims(tf.repeat(tf.expand_dims(tf.range(x_S-1), 1), x_B, 1), -1)
  tars_times = tf.reshape(target_times - tf.cast(t_asc, dtype=dtype), ((x_S-1) * a_B, 1))
  tar_inter_returns = tf.reshape(inter_returns, ((x_S-1) * a_B, 1))
  inps = tf.reshape(inp_states[:(x_S-1), :, :], ((x_S-1) * x_B, x_D))
  targets = {
    'thick_tar_states': tars_states,
    'thick_act': tars_actions,
    'thick_time': tars_times,
    'thick_latent': tars_latent,
    'thick_inter_returns': tar_inter_returns,
    'thick_stoch': tars_stoch,
  }

  # Masks for loss to ignore predictions for time steps where no gate openings followed
  #targets_mask = 1.0 - g_mask
  g_mask_reshaped_states = tf.clip_by_value(
    tf.math.reduce_sum(
      tf.cast(
        tf.reshape(g_mask_state, (int((x_S-1) * y_B), y_D)),
        tf.float32
      ), axis=-1, keepdims=True),
    0, 1,
  )
  g_mask_reshaped_actions = tf.clip_by_value(
    tf.math.reduce_sum(
      tf.cast(
        tf.reshape(g_mask_action, (int((x_S-1) * a_B), a_D)),
        tf.float32
      ), axis=-1, keepdims=True),
    0, 1,
  )
  g_mask_reshaped_times = tf.clip_by_value(
    tf.math.reduce_sum(
      tf.cast(
        tf.reshape(g_mask_time, (int((x_S-1) * y_B), 1)),
        tf.float32
      ), axis=-1, keepdims=True),
    0, 1,
  )
  loss_mask_states = 1.0 - g_mask_reshaped_states
  loss_mask_actions = 1.0 - g_mask_reshaped_actions
  loss_mask_times = 1.0 - g_mask_reshaped_times
  masks = {
    'state': loss_mask_states,
    'action': loss_mask_actions,
    'time': loss_mask_times,
  }
  return inps, targets, masks, seq_targets

