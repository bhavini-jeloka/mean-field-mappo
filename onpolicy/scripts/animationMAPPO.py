# Evaluate the agents against two different policies

import argparse
import os
from gymnasium import spaces
from onpolicy.envs.battlefield.environmentMAPPO import BattleFieldEnv, BattleField
import matplotlib.pyplot as plt
import numpy as np
import time
from onpolicy.visualizers.grid_env_renderer import GridRoomRenderer


def index2status(size, s):
        return s % size ** 2, s // size ** 2

def index2pos(size, idx):
        return np.array(np.unravel_index(idx.astype(int), (size, size)))


env = env = BattleFieldEnv()
env.reset(seed=42)
numIter = 15

size = 8
target = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

renderer = GridRoomRenderer(size, save_gif=False, save_dir=None, show_axis=True)
renderer.create_figure()
renderer.reset()

color_list =  ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
                   'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']

fixed_red_states = env.get_fixed_red()

for t in range(numIter):
    print('Time:', t)
    observation, reward, termination, truncation, info = env.last() # to truncate, all should be set to True

    blue_states = np.array([values[-1] for values in observation.values()])
    states_list = np.hstack((blue_states, fixed_red_states))

    pos_list, status_list = index2status(size, states_list)
    pos_list = index2pos(size, pos_list).T
    pos_list_tuples = [tuple(row) for row in pos_list]

    renderer.render_render_grid()
    renderer.render_agents(pos_list_tuples, status_list, color_list)
    renderer.mark_cell(target[0])
    renderer.mark_cell(target[1])
    renderer.show()
    renderer.hold(3)
        
    if termination or truncation:
        break
        # jointAction = None
    else:
        # this is where you would insert your policy
        jointAction = env.action_space(None).sample()
        
    fixed_red_states = env.get_fixed_red()
    print(fixed_red_states)

    env.step(jointAction)
    renderer.clear()
env.close()



