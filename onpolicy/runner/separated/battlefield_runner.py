    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.team_base_runner import Runner
import imageio
import matplotlib.pyplot as plt

def _t2n(x):
    return x.detach().cpu().numpy()

class BattleFieldRunner(Runner):
    def __init__(self, config):
        super(BattleFieldRunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        avg_reward_to_plot = {0: [], 1:[]}

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for team_id in range(self.num_teams):
                    self.trainer[team_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for team_id in range(self.num_teams):
                        idv_rews = []
                        for info in infos:
                            for count, info in enumerate(infos):
                                if 'individual_reward' in infos[count][team_id].keys():
                                    idv_rews.append(infos[count][team_id].get('individual_reward', 0))
                        train_infos[team_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[team_id].update({"average_episode_rewards": np.mean(self.buffer[team_id].rewards) * self.episode_length})
                        avg_reward_to_plot[team_id].append(np.mean(self.buffer[team_id].rewards))
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
            
        print('Training Ended')
        # training finishes, plot reward
        fig, ax = plt.subplots()
        
        ax.plot(self.log_interval*np.arange(len(avg_reward_to_plot[0])),avg_reward_to_plot, c='blue', label='Blue')
        ax.plot(self.log_interval*np.arange(len(avg_reward_to_plot[1])),avg_reward_to_plot, c='red', label='Red')
        ax.legend()
        ax.set_xlabel('episode')
        ax.set_ylabel('reward')
        ax.set_title('Training MAPPO for 8x8 grid')
        title = f'training result of mappo'
        plt.savefig(os.path.join(self.save_dir, title))


    # Function to concatenate the values of a dictionary into a single array
    def concatenate_dict_values(self, d):
        values = np.array(list(d.values()))
        concatenated_array = np.concatenate(values, axis=None)
        return concatenated_array
    
    def consolidate_dicts(self, list_of_dicts):
        # Initialize empty dictionaries for each key
        consolidated_dict = {0: [], 1: []}
        
        # Iterate through each dictionary in the list
        for d in list_of_dicts:
            # Append values to corresponding lists in consolidated_dict
            consolidated_dict[0].append(d[0])
            consolidated_dict[1].append(d[1])
        
        return consolidated_dict


    def warmup(self):
        # reset env
        obs = self.envs.reset()
        arr_obs = np.array([np.array(list(d.values())) for d in obs])

        share_obs = []
        for o in range(arr_obs.shape[0]):
            share_obs.append(arr_obs[o, 0, :-1])
        share_obs = np.array(share_obs)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)

        for team_id in range(self.num_teams):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, team_id]))
            self.buffer[team_id].share_obs[0] = share_obs.copy()
            self.buffer[team_id].obs[0] = np.array(list(arr_obs[:, team_id*self.num_agents: (team_id+1)*self.num_agents, :])).copy()
        

    @torch.no_grad()
    def collect(self, step):

        values = {}
        actions = {}
        actions_env = {}
        action_log_probs = {}
        rnn_states = {}
        rnn_states_critic = {}
        

        for team_id in range(self.num_teams):
            self.trainer[team_id].prep_rollout()

            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[team_id].policy.get_actions(np.concatenate(self.buffer[team_id].share_obs[step]),
                            np.concatenate(self.buffer[team_id].obs[step]),
                            np.concatenate(self.buffer[team_id].rnn_states[step]),
                            np.concatenate(self.buffer[team_id].rnn_states_critic[step]),
                            np.concatenate(self.buffer[team_id].masks[step]))
            # [agents, envs, dim]
            values[team_id] = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions[team_id] = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs[team_id] = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states[team_id] = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            rnn_states_critic[team_id] = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        arr_actions = [array.squeeze(axis=-1) for array in actions.values()]
        actions_env = np.concatenate(arr_actions, axis=1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones = self.consolidate_dicts(dones)

        for team_id in range(self.num_teams):
            mask = dones[team_id]
            num_true = np.sum(mask)

            rnn_states[team_id][mask] = np.zeros((num_true, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic[team_id][mask] = np.zeros((num_true, *self.buffer[team_id].rnn_states_critic.shape[3:]), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            masks[mask] = np.zeros((num_true, 1), dtype=np.float32)

            arr_obs = np.array([np.array(list(d.values())) for d in obs])
            share_obs = []
            for o in range(arr_obs.shape[0]):
                share_obs.append(arr_obs[o, 0, :-1])
            share_obs = np.array(share_obs)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            

            arr_rewards = np.array([np.array(list(d.values()))[team_id*self.num_agents: (team_id+1)*self.num_agents] for d in rewards])
            arr_rewards = np.expand_dims(arr_rewards, axis=-1)

            self.buffer[team_id].insert(share_obs, np.array(list(arr_obs[:, team_id*self.num_agents: (team_id+1)*self.num_agents, :])), rnn_states[team_id], rnn_states_critic[team_id], actions[team_id], action_log_probs[team_id], values[team_id], arr_rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_teams, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_teams, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for team_id in range(self.num_teams):
                self.trainer[team_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[team_id].policy.act(np.array(list(eval_obs[:, team_id])),
                                                                                eval_rnn_states[:, team_id],
                                                                                eval_masks[:, team_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[team_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[team_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[team_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[team_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[team_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, team_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_teams, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for team_id in range(self.num_teams):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, team_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % team_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_teams, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_teams, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for team_id in range(self.num_teams):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, team_id]))
                    self.trainer[team_id].prep_rollout()
                    action, rnn_state = self.trainer[team_id].policy.act(np.array(list(obs[:, team_id])),
                                                                        rnn_states[:, team_id],
                                                                        masks[:, team_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[team_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[team_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[team_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[team_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[team_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, team_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_teams, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for team_id in range(self.num_teams):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, team_id], axis=0))
                print("eval average episode rewards of agent%i: " % team_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
