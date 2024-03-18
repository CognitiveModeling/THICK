import gym
import minihack
from nle import nethack
from minihack.envs.river import MiniHackRiverNarrow, MiniHackRiver
from minihack.envs.keyroom import MiniHackKeyDoor
from minihack import MiniHackNavigation, MiniHackSkill, RewardManager

def MiniHackRegister(name):

    MOVE_ACTIONS = tuple(nethack.CompassDirection)[:4]
    APPLY_ACTIONS = tuple(list(MOVE_ACTIONS) + [nethack.Command.PICKUP, nethack.Command.APPLY])
    if 'keycorridor' in name:
        # corridor with varying length
        my_env = MiniHackKeyDoorTask(
            des_file=f"des_envs/{name}.des",
            observation_keys=['pixel_crop'],
            obs_crop_h=5,
            obs_crop_w=5,
            actions=APPLY_ACTIONS,
            character="rog-hum-new-mal",
            max_episode_steps=200,
        )
        return my_env
    elif name == 'river':
        my_env = MiniHackRiver(
            observation_keys=['pixel_crop'],
            actions=MOVE_ACTIONS,
            character="arc-hum-new-mal",
            obs_crop_h=5,
            obs_crop_w=5,
            max_episode_steps=400,
        )
        return my_env
    elif name == 'keyroomfixed5':
      my_env = gym.make(
        "MiniHack-KeyRoom-Fixed-S5-v0",
        observation_keys=['pixel_crop'],
        obs_crop_h=5,
        obs_crop_w=5,
        actions=APPLY_ACTIONS,
        autopickup=False,
        max_episode_steps=200,
      )
      return my_env
    elif name == "miniwod":
      # environments where a wand needs to be picked up to kill a minotaur
      WAND_ACTIONS = tuple(list(MOVE_ACTIONS) + [nethack.Command.PICKUP, nethack.Command.ZAP, nethack.Command.EAT])
      my_env = MiniHackWandTask(
        des_file=f'des_envs/miniwod.des',
        observation_keys=['pixel_crop'],
        obs_crop_h=5,
        obs_crop_w=5,
        actions=WAND_ACTIONS,
        character="val-hum-new-mal",
        max_episode_steps=400,
      )
      return my_env
    elif 'escape' in name:
      my_env = EscapeRoomEnv(
        des_file= f'des_envs/{name}.des',
        observation_keys=['pixel_crop'],
        obs_crop_h=5,
        obs_crop_w=5,
        character=f'sam-hum-new-fem',
        allow_all_yn_questions=True,
        allow_all_modes=False,
        max_episode_steps=400,
      )
      return my_env
    elif 'crossing' in name:
        if 'lava' in name:
            player_char = 'cav' # purely aesthetics
            reward_lose = -0.1
        else:
            player_char = 'ran'
            reward_lose = 0.0
        my_env = MiniHackRingTask(
            des_file=f'des_envs/{name}_ring.des',
            observation_keys=['pixel_crop'],
            obs_crop_h=5,
            obs_crop_w=5,
            character=f'{player_char}-hum-new-fem',
            allow_all_yn_questions=True,
            allow_all_modes=False,
            reward_lose=reward_lose,
            reward_win=1.0,
            max_episode_steps=200,
        )
        return my_env
    else:
      assert False, f'{name} does not exist'


class MiniHackKeyDoorTask(MiniHackKeyDoor):
    def __init__(self, *args, des_file, **kwargs):
        kwargs['charachter'] = kwargs['character']  # fix typo ;)
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackWandTask(MiniHackSkill):
    """ Environments with wands, when "zap" is chosen the wand is automatically selected """
    def __init__(self, *args, des_file, **kwargs):

        # only three characters currently implemented
        char_type = kwargs['character'][:3]
        assert char_type == 'rog' or char_type == 'val' or char_type == 'cav', \
            "Only Rogue, Valkyrie or Cavemen character allowed, got " + char_type
        if char_type == 'val':
            self.wand_action = nethack.Command.EAT  # e
        elif char_type == 'cav':
            self.wand_action = nethack.Command.FIRE  # f
        elif char_type == 'rog':
            self.wand_action = nethack.Command.RUSH  # g

        actions = kwargs['actions']
        assert self.wand_action in actions, "Missing action " + str(self.wand_action) + " to use wand"
        assert nethack.Command.ZAP in actions, "Missing Command.ZAP to use wand"

        self.wand_idx = actions.index(self.wand_action)
        self.zap_idx = actions.index(nethack.Command.ZAP)

        super().__init__(*args, des_file=des_file, **kwargs)

    def reset(self):
        outs = super().reset()
        # do not start with additional items (applies for rog or val)
        while self.key_in_inventory("blindfold") or self.key_in_inventory("lamp"):
            outs = super().reset()
        return outs

    def step(self, action: int):

        # different chars have different actions to confirm zapping: rog=rush, cav=fire, val=eat
        if action == self.wand_idx:
            action = self.zap_idx  # change to ZAP

        if action == self.zap_idx:
            death_key = self.key_in_inventory("wand")
            # if wand is in the inventory
            if death_key is not None:
                # zap
                obs, reward, done, info = super().step(action)  # zap
                if done:
                    return obs, reward, done, info
                # automatically select the wand
                obs, reward, done, info = super().step(self.wand_idx)
                return obs, reward, done, info
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info


class MiniHackRingTask(MiniHackSkill):
    def __init__(self, *args, des_file, **kwargs):        # only three characters currently implemented
        kwargs['charachter'] = kwargs['character']  # fix typo of devs in KeyDoorEnv ;)
        char_type = kwargs['character'][:3]
        MOVE_ACTIONS = tuple(nethack.CompassDirection)[:4]
        RELEVANT_ACTIONS = MOVE_ACTIONS[:4] + (nethack.Command.PICKUP, nethack.Command.EAT, nethack.Command.FIRE,
                                               nethack.Command.RUSH, nethack.Command.PUTON, nethack.Command.READ)
        actions = RELEVANT_ACTIONS
        self.actions = actions
        # depending on characters the actions for selecting an item index is different
        if char_type == 'val':
            # action e, f, g for item 1, 2, 3 respectively
            self._item_idxs = [actions.index(nethack.Command.EAT),
                               actions.index(nethack.Command.FIRE),
                               actions.index(nethack.Command.RUSH)]
        elif char_type == 'cav' or char_type == 'sam':
            # action f, g for item 1 or 2, only works in 2 item envs
            self._item_idxs = [actions.index(nethack.Command.FIRE), actions.index(nethack.Command.RUSH), None]
        elif char_type == 'rog' or char_type == 'ran':
            # action g for item 1, does not work for more items
            self._item_idxs = [actions.index(nethack.Command.RUSH), None, None]
        else:
            raise NotImplementedError(kwargs['character'])

        self._wearing_smth = False

        self.pickup_idx = actions.index(nethack.Command.PICKUP)
        self.puton_idx = actions.index(nethack.Command.PUTON)
        self.right_idx = actions.index(nethack.Command.READ)

        kwargs['actions'] = self.actions
        self._int_env = MiniHackKeyDoor(*args, des_file=des_file, **kwargs)

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = self._int_env.observation_space

    def reset(self, *args, **kwargs):
        outs = self._int_env.reset()
        self._wearing_smth = False
        while self._int_env.key_in_inventory("lamp") or self._int_env.key_in_inventory("blindfold"):
            # we do not want random starting items to mess up the inventory order
            outs = self._int_env.reset()
        return outs

    def step(self, action: int):
        if action == self.pickup_idx:  # picking up
            obs, reward, done, info = self._int_env.step(action)
            # Pick up is done check if there is a need to wear it
            if not self._wearing_smth and self._int_env.key_in_inventory("ring"): # wearables are directly put on
                 if not done:
                        obs, reward, done, info = self._int_env.step(self.puton_idx)
                        if not done:
                            obs, reward, done, info = self._int_env.step(self._item_idxs[0])
                            if not done:
                                obs, reward, done, info = self._int_env.step(self.right_idx)
                                self._wearing_smth = True
        else:
            obs, reward, done, info = self._int_env.step(action)
        return obs, reward, done, info

class EscapeRoomEnv:
    def __init__(self, *args, des_file, **kwargs):

        kwargs['charachter'] = kwargs['character']  # fix typo of devs in KeyDoorEnv ;)
        char_type = kwargs['character'][:3]
        MOVE_ACTIONS = tuple(nethack.CompassDirection)[:4]
        RELEVANT_ACTIONS = MOVE_ACTIONS[:4] + (nethack.Command.PICKUP, nethack.Command.APPLY, nethack.Command.DROP,
                                               nethack.Command.ZAP, nethack.Command.EAT, nethack.Command.FIRE,
                                               nethack.Command.RUSH, nethack.Command.PUTON, nethack.Command.READ,
                                               nethack.Command.REMOVE)
        actions = RELEVANT_ACTIONS
        self.actions = actions

        # depending on characters the actions for selecting an item index is different
        if char_type == 'val':
            # action e, f, g for item 1, 2, 3 respectively
            self._item_idxs = [actions.index(nethack.Command.EAT),
                               actions.index(nethack.Command.FIRE),
                               actions.index(nethack.Command.RUSH)]
        elif char_type == 'cav' or char_type == 'sam':
            # action f, g for item 1 or 2, only works in 2 item envs
            self._item_idxs = [actions.index(nethack.Command.FIRE), actions.index(nethack.Command.RUSH), None]
        elif char_type == 'rog' or char_type == 'ran':
            # action g for item 1, does not work for more items
            self._item_idxs = [actions.index(nethack.Command.RUSH), None, None]
        else:
            raise NotImplementedError(kwargs['character'])

        self._curr_items = 0
        self._curr_item_order = []

        self._coords = None
        self._wearing_smth = False
        self.blocking_objs = ["wall", "closed door", "water"]
        self.blocking_objs_levitate = ["wall", "closed door", "boulder"]  # when levitating other objects are obstacles
        self.item_names = ["ring", "key", "robe"]

        self.zap_idx = actions.index(nethack.Command.ZAP)
        self.drop_idx = actions.index(nethack.Command.DROP)
        self.pickup_idx = actions.index(nethack.Command.PICKUP)
        self.apply_idx = actions.index(nethack.Command.APPLY)
        self.puton_idx = actions.index(nethack.Command.PUTON)
        self.right_idx = actions.index(nethack.Command.READ)
        self.remove_idx = actions.index(nethack.Command.REMOVE)

        kwargs['actions'] = self.actions
        self._int_env = MiniHackKeyDoor(*args, des_file=des_file, **kwargs)

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = self._int_env.observation_space

    def update_internal_coords(self, last_act, old_items, last_inventory):
        # maintain internal coordinates to know whether the agent stands on an item
        new_agent_coords = self._get_pos_coords(last_act)
        if new_agent_coords != 4:
            coords_copy = self._coords.copy()
            self._coords = self._int_env.get_neighbor_descriptions(self._int_env.last_observation)
            self._coords[4] = coords_copy[new_agent_coords]
        current_inventory = self._get_inventory()
        if last_act == self.pickup_idx:
            if current_inventory is not None and current_inventory != last_inventory:
                if old_items == 0:
                    self._coords[4] = 'floor of a room'
                else:
                    self._coords[4] = last_inventory
        elif last_act == self.apply_idx and current_inventory is not None and 'key' in current_inventory:
            for i in range(len(self._coords)):
                if self._coords[i] == 'closed door':
                    self._coords[i] = 'open door'
                    break
        elif last_act == self.drop_idx:
            if old_items > 0:
                self._coords[4] = last_inventory

    def _standing_on_item(self):
        current_tile = self._coords[4]
        for item in self.item_names:
            if item in current_tile:
                return item
        return None

    def _get_inventory(self):
        for item_name in self.item_names:
            item_key = self._int_env.key_in_inventory(item_name)
            if item_key is not None:
                return item_name
        return None

    def _get_pos_coords(self, act):
        coords = self._act_to_coords(act)
        blocked = False
        next_tile = self._coords[self._act_to_coords(act)]
        obstacles = self.blocking_objs
        if self._get_inventory() == 'ring':  # when levitating
            obstacles = self.blocking_objs_levitate
        for obj_name in obstacles:
            if obj_name in next_tile:
                blocked = True
        if blocked:
            return 4
        return coords

    def print_coords(self):
        for i in range(3):
            print(self._coords[i*3:(i*3 + 3)])

    def _act_to_coords(self, act):
        if self.actions.index(nethack.CompassCardinalDirection.N) == act:
            return 1
        elif self.actions.index(nethack.CompassCardinalDirection.E) == act:
            return 5
        elif self.actions.index(nethack.CompassCardinalDirection.S) == act:
            return 7
        elif self.actions.index(nethack.CompassCardinalDirection.W) == act:
            return 3
        else:
            return 4

    def _item_act(self, item_name):
        return self._item_idxs[self._curr_item_order.index(item_name)]

    def step(self, action: int):
        assert action in self.action_space
        old_items = self._curr_items
        last_inventory = self._get_inventory()
        obs, reward, done, info = self._internal_step(action)
        self.update_internal_coords(action, old_items, last_inventory)
        return obs, reward, done, info

    def _internal_step(self, action: int):
        done = False
        if action == self.apply_idx:  # if apply is chosen
            if self._int_env.key_in_inventory("key") is not None:
                assert "key" in self._curr_item_order, "Key not in current item order list: " + str(self._curr_item_order)
                # try to open door with key
                obs, reward, done, info = self._int_env.step(action)
                if not done:
                    obs, reward, done, info = self._int_env.step(self._item_act("key"))
                    if not done:
                        obs, reward, done, info = self._int_env.step(self._item_act("key"))
                return obs, reward, done, info
            else:
                action = self.zap_idx  # default to no effect action
        if action == self.pickup_idx:  # picking up
            standing_on_item = self._standing_on_item()

            if standing_on_item is not None:
                if standing_on_item not in self._curr_item_order:
                    self._curr_item_order.append(standing_on_item)
                if self._curr_items == 0:  # first item is just picked up
                    self._curr_items += 1
                    obs, reward, done, info = self._int_env.step(action)
                else:  # second item results in dropping the first one
                    inventory = self._get_inventory()  # already have an item
                    if self._wearing_smth:  # remove it first
                        self._wearing_smth = False
                        obs, reward, done, info = self._int_env.step(self.remove_idx)
                    inventory = self._get_inventory()  # already have an item
                    if not done:
                        obs, reward, done, info = self._int_env.step(action)  # pick up new item
                        if not done:
                            obs, reward, done, info = self._int_env.step(self.drop_idx)  # drop old item
                            if not done:
                                obs, reward, done, info = self._int_env.step(self._item_act(inventory))  # select item
            else:
                obs, reward, done, info = self._int_env.step(action)
            # Pick up is done check if there is a need to wear it
            if not self._wearing_smth and self._int_env.key_in_inventory("ring"):  # wearables are directly put on
                 if not done:
                        obs, reward, done, info = self._int_env.step(self.puton_idx)
                        if not done:
                            obs, reward, done, info = self._int_env.step(self._item_act("ring"))
                            if not done:
                                obs, reward, done, info = self._int_env.step(self.right_idx)
                                self._wearing_smth = True
        else:
            obs, reward, done, info = self._int_env.step(action)
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        outs = self._int_env.reset()
        self._curr_items = 0
        self._curr_item_order = []
        self.item_num = 0
        self._wearing_smth = False
        while self._int_env.key_in_inventory("lamp") or self._int_env.key_in_inventory("blindfold"):
            # we do not want random starting items to mess up the inventory order
            outs = self._int_env.reset()
        self._coords = self._int_env.get_neighbor_descriptions(self._int_env.last_observation)
        self._coords[4] = "stairs up"
        return outs
