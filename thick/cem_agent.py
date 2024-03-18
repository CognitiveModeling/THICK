import numpy as np
import tensorflow as tf
import common

class CEMAgent:

    def __init__(self, config, action_dim, min_action=-1.0, max_action=1.0, bonus_factor=0.0):

        # basic configs
        self.config = config

        # action space
        self.action_dim = action_dim

        self.min_action = min_action
        self.max_action = max_action
        self.bonus_factor = bonus_factor
        self.reward_factor = config.cem.reward_factor
        self.shift_elites = None
        self.planrewnorm = common.StreamNorm(**self.config.plan_reward_norm)
        self.bonusrewnorm = common.StreamNorm(**self.config.plan_reward_norm)

    def sum_imagined_rewards(self, prior, model, reward_model, bonus_reward_model):
        feats = model.get_feat(prior)
        reward_preds = reward_model(feats).mode()
        assert len(reward_preds.shape) == 2
        if self.config.cem.discount_rewards:
            inc_time = tf.repeat(tf.expand_dims(tf.range(0, reward_preds.shape[1]), 0), reward_preds.shape[0], 0)
            disc = tf.ones_like(reward_preds) * self.config.discount
            weight = tf.pow(disc, tf.cast(inc_time, dtype=disc.dtype))
            reward_preds = weight * reward_preds
        else:
            weight = tf.ones_like(reward_preds)
        mean_rewards = tf.math.reduce_mean(reward_preds, 1)  # sum over time

        if bonus_reward_model is not None:
            reward_bonus = weight * bonus_reward_model(prior)
            mean_reward_bonus = tf.math.reduce_mean(reward_bonus, 1)  # sum over time
        else:
            mean_reward_bonus = tf.zeros_like(mean_rewards)

        _, _ = self.planrewnorm(mean_rewards)
        _, _ = self.bonusrewnorm(mean_reward_bonus)
        return self.reward_factor * mean_rewards + self.bonus_factor * mean_reward_bonus

    def get_mets(self):
        planmets = self.planrewnorm.get_mean()
        mets1 = {f'plan_exp_reward_{k}': v for k, v in planmets.items()}
        bonusmets = self.bonusrewnorm.get_mean()
        mets2 = {f'plan_exp_bonus_{k}': v for k, v in bonusmets.items()}
        return {**mets1, **mets2}

    def plan(self, model, reward_model, model_state, bonus_reward_model=None):

        action_shape = tuple([self.action_dim])
        original_batch, state_dim = model_state['deter'].shape
        initial_state = tf.nest.map_structure(
            lambda tensor: tf.tile(tensor, [self.config.cem.amount] + [1] * (tensor.shape.ndims - 1)),
            model_state,
        )
        extended_batch = initial_state['deter'].shape[0]

        if self.shift_elites is None:
            keep_elites = tf.random.normal((original_batch,
                                            self.config.cem.keep_elites,
                                            self.config.cem.horizon) + action_shape)
        else:
            keep_elites = self.shift_elites

        def iteration(mean_and_stddev, _):
            mean, stddev, keep_elites = mean_and_stddev
            # Sample action proposals from belief.
            normal = tf.random.normal((original_batch,
                                       self.config.cem.amount - self.config.cem.keep_elites,
                                       self.config.cem.horizon) + action_shape)
            action = normal * stddev[:, None] + mean[:, None]
            # append last_elites
            if self.config.cem.keep_elites > 0:
                assert len(keep_elites.shape) == 4
                action = tf.concat([action, keep_elites], 1)
                assert action.shape[1] == self.config.cem.amount

            action = tf.clip_by_value(action, self.min_action, self.max_action)
            # evaluate proposal actions.
            action = tf.reshape(action, (extended_batch, self.config.cem.horizon) + action_shape)
            preds = model.imagine(action=action, state=initial_state)
            sum_rewards = self.sum_imagined_rewards(preds, model, reward_model, bonus_reward_model)
            sum_rewards = tf.reshape(sum_rewards, (original_batch, self.config.cem.amount))
            # re-fit belief to the best ones.
            _, indices = tf.nn.top_k(sum_rewards, self.config.cem.topk, sorted=False)
            indices += tf.range(original_batch)[:, None] * self.config.cem.amount

            # do the same with keep_elites
            if self.config.cem.keep_elites > 0:
                _, keep_elites_indices = tf.nn.top_k(sum_rewards, self.config.cem.keep_elites, sorted=False)
                keep_elites = tf.gather(action, keep_elites_indices)

            best_actions = tf.gather(action, indices)
            mean, variance = tf.nn.moments(best_actions, 1)
            stddev = tf.sqrt(variance + 1e-6)
            return mean, stddev, keep_elites
        mean = tf.zeros((original_batch, self.config.cem.horizon) + action_shape)
        stddev = tf.ones((original_batch, self.config.cem.horizon) + action_shape)
        if self.config.cem.iterations < 1:
            return mean
        mean, stddev, keep_elites = tf.nest.map_structure(tf.stop_gradient, tf.scan(iteration, tf.range(self.config.cem.iterations), (mean, stddev, keep_elites)))
        if self.config.cem.keep_elites > 0 and self.config.cem.shift_elites:
            shift = tf.random.normal((original_batch, self.config.cem.keep_elites, 1) + action_shape)
            self.shift_elites = tf.concat([keep_elites[-1, :, :, 1:, :], shift], 2)
        mean, stddev = mean[-1], stddev[-1]  # select belief at last iterations.
        return mean[:, 0, :]  # only take first action

    def reset(self):
        self.shift_elites = None
