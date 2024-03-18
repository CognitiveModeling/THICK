import pyglet
import gym
from gym_minigrid.wrappers import *

def MiniGridRegister(name):
    pyglet.options['headless'] = True
    if name == '1room':
        return gym.make('MiniGrid-Empty-16x16-v0')
    elif name == 'Smallroom':
        return gym.make('MiniGrid-Empty-5x5-v0')
    elif name == 'Smallroom6':
        return gym.make('MiniGrid-Empty-6x6-v0')
    elif name == '4room':
        return gym.make('MiniGrid-FourRooms-v0')
    elif name == 'Keyroom':
        return gym.make('MiniGrid-DoorKey-16x16-v0')
    elif name == 'Keyroom8':
        return gym.make('MiniGrid-DoorKey-8x8-v0')
    elif name == 'Keyroom6':
        return gym.make('MiniGrid-DoorKey-6x6-v0')
    elif name == 'Memory13':
        return gym.make('MiniGrid-MemoryS13-v0')
    elif name == 'Obstacleroom':
        return gym.make('MiniGrid-Dynamic-Obstacles-Random-6x6-v0')
    else:
        assert False, f'{name} does not exist'
