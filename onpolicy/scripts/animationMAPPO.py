# Evaluate the agents against two different policies

import argparse
import os
import sys
from gymnasium import spaces
from onpolicy.envs.battlefield.environmentMAPPO import BattleFieldEnv, BattleField
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.visualizers.grid_env_renderer import GridRoomRenderer


def index2status(size, s):
        return s % size ** 2, s // size ** 2

def index2pos(size, idx):
        return np.array(np.unravel_index(idx.astype(int), (size, size)))

def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int,
                        default=20, help="number of players")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # current run dir
    if run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    model_dir = run_dir / 'models'

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = BattleFieldEnv()
    envs.set_seed(all_args.seed)
    obs = envs.reset()
    print('finish making environment')
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.battlefield_runner import BattleFieldRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.restore(model_dir)

    numIter = 15
    size = 8
    target = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

    renderer = GridRoomRenderer(size, save_gif=False, save_dir=None, show_axis=True)
    renderer.create_figure()
    renderer.reset()

    rnn_states = np.zeros((1, * runner.buffer.rnn_states.shape[2:]), dtype=np.float32)
    masks = np.ones((1, num_agents, 1), dtype=np.float32)

    color_list =  ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
                    'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']

    fixed_red_states = envs.get_fixed_red()

    for t in range(numIter):
        print('Time:', t)
        print(obs)
        blue_states = np.array([values[-1] for values in obs.values()])
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

        action, rnn_states = runner.policy.actor(np.concatenate(obs),
                                                np.concatenate(rnn_states),
                                                np.concatenate(masks),
                                                deterministic=True)
        actions = np.array(np.split(runner._t2n(action), 1))
        rnn_states = np.array(np.split(runner._t2n(rnn_states), 1))

        jointAction = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
            
        obs, rewards, dones, infos = envs.step(jointAction)
        renderer.clear()
        envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])