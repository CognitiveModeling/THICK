import re
import common
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

class EnsembleRSSM(common.Module):

  def __init__(
          self,
          ensemble=5,
          stoch=30,
          deter=200,
          hidden=200,
          discrete=False,
          act='elu',
          norm='none',
          std_act='softplus',
          min_std=0.1,
          rnn_norm='layer_norm',
  ):
    super().__init__()

    if discrete < 0:
      discrete = False

    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std

    self._rnn_type = rnn_type
    self._cell = GRUCell(self._deter, norm=(rnn_norm == 'layer_norm'))
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          gates=tf.zeros([batch_size, self._deter], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          gates=tf.zeros([batch_size, self._deter], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))

    return state

  def deter_dim(self):
    return self._deter


  @tf.function
  def observe(self, embed, action, is_first, state=None, rnn_sample=True):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    post, prior = common.static_scan(
      lambda prev, inputs: self.obs_step(prev[0], *inputs),
      (swap(action), swap(embed), swap(is_first)),
      (state, state),
    )
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None, rnn_sample=True):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = common.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, rnn_sample=True):

    # if is_first.any():
    prev_state, prev_action = tf.nest.map_structure(
        lambda x: tf.einsum(
            'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
        (prev_state, prev_action))
    prior = self.img_step(prev_state, prev_action, sample, rnn_sample=rnn_sample)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    x = self.get('obs_out_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], 'gates': prior['gates'],  **stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True, rnn_sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    deter = [prev_state['deter']]
    x, deter, gates = self._cell(x, deter, sample=rnn_sample)
    deter = deter[0]  # Keras wraps the state in a list.
    stats = self._suff_stats_ensemble(x)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    prior = {'stoch': stoch, 'deter': deter, 'gates': gates, **stats}
    return prior

  def _suff_stats_ensemble(self, inp):
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
      x = self.get(f'img_out_norm_{k}', NormLayer, self._norm)(x)
      x = self._act(x)
      stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
    stats = {
        k: tf.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg, prefix=''):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class ContextRSSM(common.Module):

  def __init__(
          self,
          ensemble=5,
          stoch=30,
          deter=200,
          context=16,
          hidden=200,
          discrete=False,
          act='elu',
          norm='none',
          std_act='softplus',
          min_std=0.1,
          rnn_norm='layer_norm',
          ctxt_rnn_type='SimpleGateL0RD',
          ctxt_sample_noise=0.05,
          ctxt_rnn_out_size=-1,
          ctxt_always_sample=False,
          act_dim=None,
  ):
    super().__init__()

    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._context = context
    self._hidden = hidden
    self._discrete = discrete
    self._act = get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    self.act_dim = act_dim

    # RNN learning deter hidden states
    self._cell = GRUCell(self._deter, norm=(rnn_norm == 'layer_norm'))

    if ctxt_rnn_type == 'SimpleGateL0RD':
      # GateL0RD embedding context memory
      self._ctxt_rnn = SimpleGateL0RDCell(
        self._context,
        sample_sd=ctxt_sample_noise,
        out_size=ctxt_rnn_out_size,
        always_sample=ctxt_always_sample,
        headless=True,
      )
    else:
      raise NotImplementedError(ctxt_rnn_type)

    self._ctxt_head = None
    self._ctxt_head = MLP(self._deter, layers=3, units=hidden, name='contextout')

    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          gates=tf.zeros([batch_size, self._deter], dtype),
          context=self._ctxt_rnn.get_initial_state(None, batch_size, dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype),
          ctxt_logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          ctxt_stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          prev_stoch=tf.zeros([batch_size, self._stoch*self._discrete], dtype),
          prev_action=tf.zeros([batch_size, self.act_dim], dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          gates=tf.zeros([batch_size, self._deter], dtype),
          context=self._ctxt_rnn.get_initial_state(None, batch_size, dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype),
          ctxt_mean=tf.zeros([batch_size, self._stoch], dtype),
          ctxt_std=tf.zeros([batch_size, self._stoch], dtype),
          ctxt_stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          prev_stoch=tf.zeros([batch_size, self._stoch*self._discrete], dtype),
          prev_action=tf.zeros([batch_size, self.act_dim], dtype))

    return state

  def deter_dim(self):
    return self._deter

  def ctxt_dim(self):
    return self._context

  def get_feat_dim(self):
    if self._discrete:
      stoch_dim = self._stoch * self._discrete
    else:
      stoch_dim = self._stoch
    return stoch_dim + self._deter + self._context

  @tf.function
  def observe(self, embed, action, is_first, state=None, rnn_sample=True):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))

    if state is None:
      state = self.initial(tf.shape(action)[0])

    obs_step_sample = lambda a,b: self.obs_step(a, *b, rnn_sample=rnn_sample)
    post, prior = common.static_scan(
      lambda prev, inputs: obs_step_sample(prev[0], inputs),
      (swap(action), swap(embed), swap(is_first)),
      (state, state),
    )
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None, rnn_sample=True):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    img_step_sample = lambda a,b: self.img_step(a, b, rnn_sample=rnn_sample)
    prior = common.static_scan(img_step_sample, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter'], state['context']], -1)

  @tf.function
  def _get_coarse_out_internal(self, state):
    stoch = self._cast(state['prev_stoch'])
    output_inp = tf.concat([stoch, state['prev_action']], -1)
    ctxt_out = self._ctxt_head(tf.concat([output_inp, state['context']], -1)).mode()
    return tf.concat([stoch, ctxt_out], -1)

  @tf.function
  def get_coarse_pred(self, state, sample=True):
    if 'ctxt_stoch' in state:
      stoch = self._cast(state['ctxt_stoch'])
    else:
      ctxt_out = self._get_coarse_out_internal(state)
      ctxt_stats = self._suff_stats_ensemble(ctxt_out, prefix='ctxt_')
      ctxt_index = tf.random.uniform((), 0, self._ensemble, tf.int32)
      ctxt_stats = {k: v[ctxt_index] for k, v in ctxt_stats.items()}
      ctxt_dist = self.get_dist(ctxt_stats, prefix='ctxt_')
      stoch = ctxt_dist.sample() if sample else ctxt_dist.mode()
    if self._discrete:
        shape = stoch.shape[:-2] + [self._stoch * self._discrete]
        stoch = tf.reshape(stoch, shape)
    ctxt = state['context']
    return tf.concat([stoch, ctxt], -1)

  def get_last_coarse_pred(self, state):
    assert 'prev_context' in state
    assert 'prev_stoch' in state
    stoch = self._cast(state['prev_stoch'])
    ctxt = state['prev_context']
    return tf.concat([stoch, ctxt], -1)

  @tf.function
  def get_coarse_out(self, state):
    if 'stoch' in state:
      stoch = self._cast(state['stoch'])
    else:
      assert False, "Stoch not found " + str(state)
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    ctxt = state['context']
    return tf.concat([stoch, ctxt], -1)

  def get_dist(self, state, prefix=''):
    # prefix determines if context-based dist or regular dist
    if self._discrete:
      logit = state[f'{prefix}logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state[f'{prefix}mean'], state[f'{prefix}std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, rnn_sample=True):

    # if is_first.any():
    prev_state, prev_action = tf.nest.map_structure(
        lambda x: tf.einsum(
            'b,b...->b...', 1.0 - is_first.astype(x.dtype), x),
        (prev_state, prev_action))

    prior = self.img_step(prev_state, prev_action, sample, rnn_sample=rnn_sample)

    x = tf.concat([prior['context'], prior['deter'], embed], -1)
    x = self.get('obs_out', tfkl.Dense, self._hidden)(x)
    x = self.get('obs_out_norm', NormLayer, self._norm)(x)
    x = self._act(x)
    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()
    ctxt_stats = {f'ctxt_{k}': v for k, v in stats.items()}
    ctxt_stats['ctxt_stoch'] = dist.sample() if sample else dist.mode()
    post = {'stoch': stoch, 'deter': prior['deter'], 'context': prior['context'],
            'gates': prior['gates'], 'prev_stoch': prior['prev_stoch'], 'prev_action':prior['prev_action'],
            **stats, **ctxt_stats}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True, rnn_sample=True):
    prev_stoch = self._cast(prev_state['stoch'])
    prev_action = self._cast(prev_action)
    if self._discrete:
      shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
      prev_stoch = tf.reshape(prev_stoch, shape)
    x = tf.concat([prev_stoch, prev_action], -1)
    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', NormLayer, self._norm)(x)
    x = self._act(x)

    context = [prev_state['context']]
    _, context, gates = self._ctxt_rnn(x, context, sample=rnn_sample)
    context = context[0]

    x = tf.concat([x, context], -1)
    deter = [prev_state['deter']]
    x, deter, _ = self._cell(x, deter, sample=rnn_sample)
    deter = deter[0]  # Keras wraps the state in a list.
    x = tf.concat([x, context], -1)

    stats = self._suff_stats_ensemble(x)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stoch = dist.sample() if sample else dist.mode()

    ctxt_dict = {'stoch': stoch, 'context': context, 'prev_stoch': prev_stoch, 'prev_action':prev_action}
    ctxt_out = self._get_coarse_out_internal(ctxt_dict)
    ctxt_stats = self._suff_stats_ensemble(ctxt_out, prefix='ctxt_')
    ctxt_index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    ctxt_stats = {k: v[ctxt_index] for k, v in ctxt_stats.items()}
    ctxt_dist = self.get_dist(ctxt_stats, prefix='ctxt_')
    ctxt_stoch = ctxt_dist.sample() if sample else ctxt_dist.mode()
    ctxt_stats['ctxt_stoch'] = ctxt_stoch
    ctxt_stats['prev_stoch'] = ctxt_dict['prev_stoch']
    ctxt_stats['prev_action'] = ctxt_dict['prev_action']

    prior = {'stoch': stoch, 'deter': deter, 'context': context, 'gates': gates, **stats, **ctxt_stats}
    return prior

  def _suff_stats_ensemble(self, inp, prefix=''):
    # prefix determines if context-based stats or regular stats
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self.get(f'{prefix}img_out_{k}', tfkl.Dense, self._hidden)(inp)
      x = self.get(f'{prefix}img_out_norm_{k}', NormLayer, self._norm)(x)
      x = self._act(x)
      stats.append(self._suff_stats_layer(f'{prefix}img_dist_{k}', x, prefix=prefix))
    stats = {
        k: tf.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x, prefix=''):
    # prefix determines if context-based stats or regular stats
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {f'{prefix}logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {f'{prefix}mean': mean, f'{prefix}std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg, prefix=''):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs, prefix=prefix), self.get_dist(rhs, prefix=prefix))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs, prefix=prefix), self.get_dist(sg(rhs), prefix=prefix))
      value_rhs = kld(self.get_dist(sg(lhs), prefix=prefix), self.get_dist(rhs, prefix=prefix))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value


class Encoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
    self.shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers

  @tf.function
  def __call__(self, data):
    key, shape = list(self.shapes.items())[0]
    batch_dims = data[key].shape[:-len(shape)]
    data = {
        k: tf.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_keys:
      outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
    if self.mlp_keys:
      outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
    output = tf.concat(outputs, -1)
    return output.reshape(batch_dims + output.shape[1:])

  def _cnn(self, data):
    x = tf.concat(list(data.values()), -1)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, kernel in enumerate(self._cnn_kernels):
      depth = 2 ** i * self._cnn_depth
      x = self.get(f'conv{i}', tfkl.Conv2D, depth, kernel, 2)(x)
      x = self.get(f'convnorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x.reshape(tuple(x.shape[:-3]) + (-1,))

  def _mlp(self, data):
    x = tf.concat(list(data.values()), -1)
    x = x.astype(prec.global_policy().compute_dtype)
    for i, width in enumerate(self._mlp_layers):
      x = self.get(f'dense{i}', tfkl.Dense, width)(x)
      x = self.get(f'densenorm{i}', NormLayer, self._norm)(x)
      x = self._act(x)
    return x


class Decoder(common.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
      cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400], name=''):
    self._shapes = shapes
    self.cnn_keys = [
        k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
    self.mlp_keys = [
        k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
    self._act = get_act(act)
    self._norm = norm
    self._cnn_depth = cnn_depth
    self._cnn_kernels = cnn_kernels
    self._mlp_layers = mlp_layers
    self._name = name

  def __call__(self, features):
    features = tf.cast(features, prec.global_policy().compute_dtype)
    outputs = {}
    if self.cnn_keys:
      outputs.update(self._cnn(features))
    if self.mlp_keys:
      outputs.update(self._mlp(features))
    return outputs

  def _cnn(self, features):
    channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
    ConvT = tfkl.Conv2DTranspose
    x = self.get('convin' + self._name, tfkl.Dense, 32 * self._cnn_depth)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._cnn_depth])
    for i, kernel in enumerate(self._cnn_kernels):
      depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
      act, norm = self._act, self._norm
      if i == len(self._cnn_kernels) - 1:
        depth, act, norm = sum(channels.values()), tf.identity, 'none'
      x = self.get(f'conv{i}' + self._name, ConvT, depth, kernel, 2)(x)
      x = self.get(f'convnorm{i}' + self._name, NormLayer, norm)(x)
      x = act(x)
    x = x.reshape(features.shape[:-1] + x.shape[1:])
    means = tf.split(x, list(channels.values()), -1)
    dists = {
        key: tfd.Independent(tfd.Normal(mean, 1), 3)
        for (key, shape), mean in zip(channels.items(), means)}
    return dists

  def _mlp(self, features):
    shapes = {k: self._shapes[k] for k in self.mlp_keys}
    x = features
    for i, width in enumerate(self._mlp_layers):
      x = self.get(f'dense{i}' + self._name, tfkl.Dense, width)(x)
      x = self.get(f'densenorm{i}' + self._name, NormLayer, self._norm)(x)
      x = self._act(x)
    dists = {}
    for key, shape in shapes.items():
      dists[key] = self.get(f'dense_{key}' + self._name, DistLayer, shape)(x)
    return dists


class MLP(common.Module):

  def __init__(self, shape, layers, units, act='elu', norm='none', name='', **out):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = get_act(act)
    self._out = out
    self._name = name

  def __call__(self, features):
    x = tf.cast(features, prec.global_policy().compute_dtype)
    x = x.reshape([-1, x.shape[-1]])
    for index in range(self._layers):
      x = self.get(f'dense{index}' + self._name, tfkl.Dense, self._units)(x)
      x = self.get(f'norm{index}'+ self._name, NormLayer, self._norm)(x)
      x = self._act(x)
    x = x.reshape(features.shape[:-1] + [x.shape[-1]])
    return self.get(f'out{self._name}', DistLayer, self._shape, **self._out)(x)

class MultipGate(common.Module):

  def __init__(self, size, act='tanh', **kwargs):
    self._size = size
    self._act = get_act(act)
    self._o_layer = tfkl.Dense(size, use_bias=True, **kwargs)
    self._p_layer = tfkl.Dense(size, use_bias=True, **kwargs)

  def __call__(self, inps):
    p_out = self._p_layer(inps)
    o_out = self._o_layer(inps)
    return self._act(p_out) * tf.nn.sigmoid(o_out)

class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act='tanh', update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = get_act(act)
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state, sample):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    gates = tf.stop_gradient(update)
    return output, [output], tf.cast(gates, tf.float32)

class SimpleGateL0RDCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, act='tanh', sample_sd=0.05, out_size=-1, always_sample=False, headless=False, **kwargs):
    super().__init__()

    self._size = size
    self._act = get_act(act)

    if out_size <= 0:
      self._out_size = size
    else:
      self._out_size = out_size
    self._always_sample = always_sample
    self._headless = headless

    self._r_layer = tfkl.Dense(self._size, use_bias=True, **kwargs)
    self._g_layer = tfkl.Dense(self._size, use_bias=True, **kwargs)
    if not self._headless:
      self._o_layer = tfkl.Dense(self._out_size, use_bias=True, **kwargs)
      self._p_layer = tfkl.Dense(self._out_size, use_bias=True, **kwargs)
    self._sample_sd = sample_sd

  @property
  def state_size(self):
    return self._size

  @tf.function
  def call(self, inputs, state, sample):
    state = state[0]  # Keras wraps the state in a list.
    overall_inp = tf.concat([inputs, state], -1)

    # candidate hidden state
    cand = self._r_layer(overall_inp)
    cand = self._act(cand)
    update = self._g_layer(overall_inp)
    dtype = update.dtype

    if sample or self._always_sample:
      gate_noise = tf.random.normal(tf.shape(update), 0.0, self._sample_sd, dtype=dtype)
      update_gate = tf.nn.relu(tf.nn.tanh(update + gate_noise))
    else:
      update_gate = tf.nn.relu(tf.nn.tanh(update))

    # update latent state
    next_state = update_gate * cand + (1 - update_gate) * state

    # binarized gate activation with straight-through estimator
    prec_update_gate = tf.cast(update_gate, tf.float32)  # always high precision for exact gate regularization
    gates = tf.stop_gradient(tf.math.ceil(prec_update_gate)) + (prec_update_gate - tf.stop_gradient(prec_update_gate))

    if self._headless:
      output = next_state
    else:
      updated_overall = tf.concat([inputs, next_state], -1)
      p_out = self._p_layer(updated_overall)
      o_out = self._o_layer(updated_overall)
      output = self._act(p_out) * tf.nn.sigmoid(o_out)

    return output, [next_state], gates


class DistLayer(common.Module):

  def __init__(
      self, shape, dist='mse', min_std=0.1, init_std=0.0, name=''):
    self._shape = shape
    self._dist = dist
    self._min_std = min_std
    self._init_std = init_std
    self._name = name

  def __call__(self, inputs):
    out = self.get(f'out{self._name}', tfkl.Dense, np.prod(self._shape))(inputs)
    out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
    out = tf.cast(out, tf.float32)
    if self._dist in ('normal', 'tanh_normal', 'trunc_normal', 'clamp_normal', 'pos_clamp_normal'):
      std = self.get(f'std{self._name}', tfkl.Dense, np.prod(self._shape))(inputs)
      std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
      std = tf.cast(std, tf.float32)
    if self._dist == 'mse':
      dist = tfd.Normal(out, 1.0)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'normal':
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'clamp_normal':
      std = tf.nn.elu(std) + 1.0 + self._min_std
      dist = tfd.Normal(out, std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'pos_clamp_normal':
      std = tf.nn.elu(std) + 1.0 + self._min_std
      dist = tfd.Normal(tf.nn.relu(out), std)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'tanh_normal':
      mean = 5 * tf.tanh(out / 5)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, common.TanhBijector())
      dist = tfd.Independent(dist, len(self._shape))
      return common.SampleDist(dist)
    if self._dist == 'trunc_normal':
      std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
      dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
      return tfd.Independent(dist, 1)
    if self._dist == 'onehot':
      return common.OneHotDist(out)
    raise NotImplementedError(self._dist)


class NormLayer(common.Module):

  def __init__(self, name):
    if name == 'none':
      self._layer = None
    elif name == 'layer':
      self._layer = tfkl.LayerNormalization()
    else:
      raise NotImplementedError(name)

  def __call__(self, features):
    if not self._layer:
      return features
    return self._layer(features)


def get_act(name):
  if name == 'none':
    return tf.identity
  if name == 'mish':
    return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
  elif hasattr(tf.nn, name):
    return getattr(tf.nn, name)
  elif hasattr(tf, name):
    return getattr(tf, name)
  else:
    raise NotImplementedError(name)
