
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_teams = 2

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)


        if self.all_args.algorithm_name == "happo":
            from onpolicy.algorithms.happo.happo_trainer import HAPPO as TrainAlgo
            from onpolicy.algorithms.happo.policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "hatrpo":
            from onpolicy.algorithms.hatrpo.hatrpo_trainer import HATRPO as TrainAlgo
            from onpolicy.algorithms.hatrpo.policy import HATRPO_Policy as Policy
        else:
            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.policy = []
        for team_id in range(self.num_teams):
            share_observation_space = self.envs.share_observation_space[team_id][0] if self.use_centralized_V else self.envs.observation_space[team_id][0]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[team_id][0],
                        share_observation_space,
                        self.envs.action_space[team_id][0],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for team_id in range(self.num_teams):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[team_id], device = self.device)
            # buffer[
            share_observation_space = self.envs.share_observation_space[team_id][0] if self.use_centralized_V else self.envs.observation_space[team_id][0]
            bu = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[team_id][0],
                                        share_observation_space,
                                        self.envs.action_space[team_id][0])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for team_id in range(self.num_teams):
            self.trainer[team_id].prep_rollout()
            next_value = self.trainer[team_id].policy.get_values(np.concatenate(self.buffer[team_id].share_obs[-1]), 
                                                                np.concatenate(self.buffer[team_id].rnn_states_critic[-1]),
                                                                np.concatenate(self.buffer[team_id].masks[-1]))

            next_value = np.array(np.split(_t2n(next_value), self.n_rollout_threads))
            self.buffer[team_id].compute_returns(next_value, self.trainer[team_id].value_normalizer)

    def train(self):
        train_infos = {}

        for team_id in torch.randperm(self.num_teams):
            self.trainer[team_id].prep_training()
            train_infos[team_id] = self.trainer[team_id].train(self.buffer[team_id])      
            self.buffer[team_id].after_update()
        return train_infos

    def save(self):
        for team_id in range(self.num_teams):
            policy_actor = self.trainer[team_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(team_id) + ".pt")
            policy_critic = self.trainer[team_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(team_id) + ".pt")
            if self.trainer[team_id]._use_valuenorm:
                policy_vnrom = self.trainer[team_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(team_id) + ".pt")

    def restore(self):
        for team_id in range(self.num_teams):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(team_id) + '.pt')
            self.policy[team_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(team_id) + '.pt')
            self.policy[team_id].critic.load_state_dict(policy_critic_state_dict)
            if self.trainer[team_id]._use_valuenorm:
                policy_vnrom_state_dict = torch.load(str(self.model_dir) + '/vnrom_agent' + str(team_id) + '.pt')
                self.trainer[team_id].value_normalizer.load_state_dict(policy_vnrom_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for team_id, infos in train_infos.items():
            for k, v in infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
