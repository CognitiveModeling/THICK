import collections
import math
import numpy as np
import tensorflow as tf
from cem_agent import CEMAgent

from typing import Any, Dict, List, Optional
from tensorflow_probability import distributions as tfd

MAXIMUM_FLOAT_VALUE = float('inf')
KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])
SOFTMAX_MAX = 500.0  # fixed value to avoid intractable exp computations


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = tf.math.maximum(self.maximum, value)
        self.minimum = tf.math.maximum(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:

    def __init__(self, prior: float):
        self.visit_count = 0  # N
        self.prior = prior
        self.value_sum = 0.0  # Q
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0  # R
        self.discount = None  # discount

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSAction:

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class MCTSActionHistory:
    """Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[MCTSAction], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return MCTSActionHistory(self.history, self.action_space_size)

    def add_action(self, action: MCTSAction):
        self.history.append(action)

    def last_action(self) -> MCTSAction:
        return self.history[-1]

    def action_space(self) -> List[MCTSAction]:
        return [MCTSAction(i) for i in range(self.action_space_size)]


class MCTSConfig:

    def __init__(self,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 random_rollouts: int,
                 rollout_depth: int,
                 rollout_value_factor: float,
                 known_bounds: Optional[KnownBounds] = None):
        ### Self-Play
        self.num_simulations = num_simulations

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        self.random_rollouts = random_rollouts
        self.rollout_depth = rollout_depth
        self.rollout_value_factor = rollout_value_factor

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

    # We expand a node using the value, reward and policy prediction obtained from


def onehot_tf_action(action, action_dim):
    reference = np.zeros(action_dim, dtype=np.float32)
    reference[int(action.index)] = 1.0
    return tf.expand_dims(tf.convert_to_tensor(reference), 0)


# the neural network.
def expand_node(node, hidden_state, reward, actions, action_prior, discount):
    node.hidden_state = hidden_state
    node.reward = reward
    node.discount = discount
    policy = {a: action_prior[a.index] for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


# at the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + node.discount * value


def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward + child.discount * child.value())
    else:
        value_score = 0.0
    return prior_score + value_score



# yelect the child with the highest UCB score.
def select_child(config: MCTSConfig, node: Node,
                 min_max_stats: MinMaxStats):

    max_score = -MAXIMUM_FLOAT_VALUE
    best_action = None
    best_child = None
    for action, child in node.children.items():
        score = ucb_score(node, child, min_max_stats)
        if tf.math.greater_equal(score, max_score):
            max_score = score
            best_child = child
            best_action = action
    return max_score, best_action, best_child


def softmax(logits):
    bottom = sum([math.exp(np.clip(x, 0, SOFTMAX_MAX)) for x in logits])
    softmax = [math.exp(np.clip(x, 0, SOFTMAX_MAX)) / bottom for x in logits]
    return softmax


def softmax_sample(logit_action, temp=1.0):
    logits = []
    actions = []
    for i in range(len(logit_action)):
        logit_i, action_i = logit_action[i]
        logits.append(logit_i)
        actions.append(action_i)
    logits_temp = [x / temp for x in logits]
    probs = softmax(logits_temp)
    return actions[np.argmax(np.random.multinomial(1, probs))]


def select_action(node: Node):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    action = softmax_sample(visit_counts)
    return action


# at the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MCTSConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


# core Monte Carlo Tree Search algorithm.
# to decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config, action_prior, model_step, reward_pred, discount_pred, root: Node, action_history: MCTSActionHistory, min_max_stats: MinMaxStats):
    max_depth = -1
    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]
        depth = 0
        while node.expanded():
            _, action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)
            depth += 1
        parent = search_path[-2]
        state_prior = model_step(parent.hidden_state, onehot_tf_action(history.last_action(), action_dim=history.action_space_size))
        actions = action_history.action_space()
        value = reward_pred(state_prior)
        expand_node(node, hidden_state=state_prior, reward=value, actions=actions,
                    action_prior=tf.stop_gradient(action_prior(state_prior)), discount=discount_pred(state_prior))
        backpropagate(search_path, value,  min_max_stats)
        if depth > max_depth:
            max_depth = depth
    return max_depth

class DummyPlanner:
    def __init__(self, config):
        self.config = config

    def plan(self, model=None, reward_model=None, model_state=None, bonus_reward_model=None):
        raise NotImplementedError("Planning called on Dummy Planner.")

    def reset(self):
        return

class HL_MCTS_Agent:

    def __init__(self, config, action_dim):

        # basic configs
        self.config = config
        self.mcts_config = config.mcts

        # tracked over optimizations
        self.min_max_stats = MinMaxStats(None)
        self.act_history = []

        # action space
        self.action_dim = action_dim
        self.legal_moves = []
        for m in range(action_dim):
            self.legal_moves.append(MCTSAction(m))

    def plan(self, model_state, init_reward, action_prior, model_step, reward_pred, discount_pred):

        root = Node(0)
        state_val = init_reward
        val = state_val
        expand_node(node=root,
                    hidden_state=model_state,
                    discount=1.0,
                    reward=val,
                    actions=self.legal_moves,
                    action_prior=tf.stop_gradient(action_prior(model_state)))
        backpropagate([root], val, self.min_max_stats)
        add_exploration_noise(self.mcts_config, root)

        # we then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        max_d = run_mcts(
            config=self.mcts_config,
            model_step=model_step,
            action_prior=action_prior,
            reward_pred=reward_pred,
            discount_pred=discount_pred,
            root=root,
            action_history=MCTSActionHistory(self.act_history, self.action_dim),
            min_max_stats=self.min_max_stats
        )
        action = select_action(root)
        best_child_state = root.children[action].hidden_state
        self.act_history.append(action)
        return onehot_tf_action(action, action_dim=self.action_dim), best_child_state

    def reset(self):
        self.min_max_stats = MinMaxStats(None)
        self.act_history = []


# Similarity measures between latent states

def max_cosine_dist(goal, feat):
    sg_feat = tf.stop_gradient(feat)
    rep_goal = tf.repeat(tf.stop_gradient(goal), axis=0, repeats=sg_feat.shape[0])
    gnorm = tf.linalg.norm(rep_goal, axis=-1, keepdims=True) + 1e-12
    fnorm = tf.linalg.norm(sg_feat, axis=-1, keepdims=True) + 1e-12
    norm = tf.maximum(gnorm, fnorm)
    return tf.einsum('...i,...i->...', rep_goal / norm, sg_feat / norm)[:]


def max_cosine_sim_seq(goal, feat):
    sg_feat = tf.stop_gradient(feat)
    rep_goal = tf.repeat(
        tf.expand_dims(tf.repeat(tf.stop_gradient(goal), axis=0, repeats=sg_feat.shape[1]), 0),
        axis=0, repeats=sg_feat.shape[0]
    )
    gnorm = tf.linalg.norm(rep_goal, axis=-1, keepdims=True) + 1e-12
    fnorm = tf.linalg.norm(sg_feat, axis=-1, keepdims=True) + 1e-12
    norm = tf.maximum(gnorm, fnorm)
    return tf.einsum('...i,...i->...', rep_goal / norm, sg_feat / norm)[:]


def cat_prob_cosine_sim_seq(goal_logit, feat_logit):
    feat_logit = tf.stop_gradient(feat_logit)
    sg_feat_probs = tf.nn.softmax(feat_logit)
    rep_goal_probs = tf.repeat(
        tf.expand_dims(tf.repeat(tf.nn.softmax(tf.stop_gradient(goal_logit)), axis=0, repeats=sg_feat_probs.shape[1]), 0),
        axis=0, repeats=sg_feat_probs.shape[0]
    )
    return tf.math.reduce_mean(-tf.keras.losses.cosine_similarity(rep_goal_probs, sg_feat_probs), -1)


def logit_cosine_sim_seq(goal_logit, feat_logit):
    feat_logit = tf.stop_gradient(feat_logit)
    sg_feat_probs = tf.nn.softmax(feat_logit)
    sg_goal_probs = tf.stop_gradient(tf.nn.softmax(goal_logit))
    return -tf.math.reduce_mean(tf.keras.losses.cosine_similarity(sg_goal_probs, sg_feat_probs), -1)


def cat_prob_cosine_sim(goal_logit, feat_logit):
    sg_feat_probs = tf.nn.softmax(tf.stop_gradient(feat_logit))
    sg_goal_probs = tf.nn.softmax(tf.stop_gradient(goal_logit))
    return tf.math.reduce_mean(-tf.keras.losses.cosine_similarity(sg_goal_probs, sg_feat_probs), -1)


def categorical_sample_prob_seq(goal_sample, feat_logit, ll_model):
    feat_logit = tf.stop_gradient(feat_logit)
    fake_state = {'logit': feat_logit}
    dist = ll_model.get_dist(fake_state)
    rep_goal_sample = tf.repeat(
        tf.expand_dims(tf.repeat(tf.stop_gradient(goal_sample), axis=0, repeats=feat_logit.shape[1]), 0),
        axis=0, repeats=feat_logit.shape[0],
    )
    return dist.prob(rep_goal_sample)


def categorical_sample_prob(goal_sample, feat_logit, ll_model):
    feat_logit = tf.stop_gradient(feat_logit)
    fake_state = {'logit': feat_logit}
    dist = ll_model.get_dist(fake_state)
    return dist.prob(tf.stop_gradient(goal_sample))


def categorical_kl_seq(goal_logit, feat_logit, ll_model):
    feat_logit = tf.stop_gradient(feat_logit)
    fake_state = {'logit': feat_logit}
    dist = ll_model.get_dist(fake_state)
    rep_goal_logit = tf.repeat(
        tf.expand_dims(tf.repeat(tf.stop_gradient(goal_logit), axis=0, repeats=feat_logit.shape[1]), 0),
        axis=0, repeats=feat_logit.shape[0],
    )
    fake_state2 = {'logit': rep_goal_logit}
    dist2 = ll_model.get_dist(fake_state2)
    return -tfd.kl_divergence(dist, dist2)


class HierarchicalPlanner(object):

    def __init__(self, config, hl_action_dim, ll_action_dim, discrete_action_space):

        # basic configs
        self.config = config

        self.hl_planner = HL_MCTS_Agent(config, hl_action_dim)
        if discrete_action_space:
            # planning currently only for continuous low-level action spaces
            raise NotImplementedError(discrete_action_space)
        else:
            self.ll_planner = CEMAgent(config, ll_action_dim, bonus_factor=config.cem.bonus_factor)

        self.ll_action_dim = ll_action_dim

        self.goal = None
        self.first_goal = None
        self.last_goal_sequence = []
        self.inter_goals = []
        self.hl_action = None
        self.replan_next = 0

    def plan(self, hl_model, ll_model, ll_reward_model, model_state, goal):
        
        # plan when (1.) gates are open and planning is not inhibited, (2.) first step or (3.) when replanning every t
        if (tf.math.reduce_sum(model_state['gates']) > 0 and len(self.inter_goals) > self.config.hl_replan_inhibt_t) \
                or goal is None or self.config.hl_replan_every:
            self.inter_goals = []
            self.hl_planner.reset()
            model_step = lambda state, act: hl_model.hierarchical_img_step(start=state, hl_act=act, sample_stoch=self.config.mcts.state_sample)
            action_prior = lambda state: hl_model.hl_action_prior(state)
            reward_pred = lambda state: state['discount'] * state['mean_reward'] + state['inter_reward']
            discount_pred = lambda state: state['discount']
            init_reward = tf.zeros(model_state['deter'].shape[:-1], model_state['deter'].dtype)
            self.hl_action, goal = self.hl_planner.plan(model_state, init_reward=init_reward,
                                                        model_step=model_step, action_prior=action_prior,
                                                        reward_pred=reward_pred, discount_pred=discount_pred)
            if self.goal is None:
                self.first_goal = goal
            self.goal = goal
            self.inter_goals.append(goal)
        else:
            self.inter_goals.append(goal)

        self.last_goal_sequence.append(goal)
        goal_prefix = ''
        if self.config.hl_plan_pre_ll_goal:
            goal_prefix = 'pre_ll_'
        if self.config.hl_goal_distance == 'stoch':
            # maximize max cosine similarity to subgoal
            reward_bonus_fn = lambda seq: max_cosine_sim_seq(goal=goal[f'{goal_prefix}stoch'], feat=seq['stoch'])
        elif self.config.hl_goal_distance == 'logit':
            reward_bonus_fn = lambda seq: cat_prob_cosine_sim_seq(goal_logit=goal[f'{goal_prefix}logit'], 
                                                                  feat_logit=seq['logit'])
        elif self.config.hl_goal_distance == 'prob':
            reward_bonus_fn = lambda seq: categorical_sample_prob_seq(goal_sample=goal[f'{goal_prefix}stoch'], 
                                                                      feat_logit=seq['logit'], ll_model=ll_model)
        elif self.config.hl_goal_distance == 'kl':
                reward_bonus_fn = lambda seq: categorical_kl_seq(goal_logit=goal[f'{goal_prefix}logit'], 
                                                                 feat_logit=seq['logit'], ll_model=ll_model)
        else:
            raise NotImplementedError(self.config.hl_goal_distance)
        ll_action = self.ll_planner.plan(
                model=ll_model, reward_model=ll_reward_model,
                model_state=model_state, bonus_reward_model=reward_bonus_fn,
        )

        return ll_action, self.hl_action, goal

    def viz_first_goal(self, agent):
        img = agent.wm.ctxt_heads['decoder'](agent.wm.rssm.get_coarse_pred(self.first_goal))['image'].mode() + 0.5
        self.first_goal = None
        return img

    def viz_last_goal_seq(self, agent):
        img_seq = []
        for g in self.last_goal_sequence:
            img = agent.wm.ctxt_heads['decoder'](agent.wm.rssm.get_coarse_pred(g))['image'].mode() + 0.5
            img_seq.append(img)
        self.last_goal_sequence = []
        video = tf.stack(img_seq, 0)
        T, B, H, W, C = video.shape
        return video.reshape((T*B, H, W, C))

    def get_mets(self):
        return self.ll_planner.get_mets()

    def reset(self):
        self.goal = None
        self.hl_action = None
        self.first_goal = None
        self.last_goal_sequence = []
        self.inter_goals = []
