import numpy as np
from typing import Tuple
from sklearn.preprocessing import normalize
import time


class mfOracle:
    def __init__(self, n_blue_states: int, n_blue_actions: int, n_red_states: int, n_red_actions: int,
                 n_blue_agents: int, n_red_agents: int, target_pos):

        self.n_blue_states, self.n_blue_actions = n_blue_states, n_blue_actions # n_blue_states = self.size**2 x 2 {0, 1} ---- first self.size**2 are alive, the second set are dead
        self.n_red_states, self.n_red_actions = n_red_states, n_red_actions
        self.n_blue_agents, self.n_red_agents = n_blue_agents, n_red_agents
        self.rho = self.n_blue_agents / (self.n_blue_agents + n_red_agents)
        self.size = int((n_blue_states/2)**0.5)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0])
        }

        self.grid = np.arange(self.size*self.size).reshape(self.size, self.size)

        self.alpha_x, self.alpha_y = 20, 20
        self.beta_x, self.beta_y = 0, 0

        self.target_indices = []

        for i in range(target_pos.shape[0]):
            self.target_indices.append(self.pos2index(target_pos[i, :]))

        self.generate_neighbour_matrix()
        self.generate_position_transition_matrix()

    def pos2index(self, pos):
        return np.ravel_multi_index(pos, (self.size, self.size))
    
    def index2pos(self, idx):
        return np.array(np.unravel_index(idx, (self.size, self.size)))
    
    def neighbourhood(self, state):
        pos = self.index2pos(state)
        indices = []
        '''
        # Calculate the range for row and column indices
        
        row_start = max(0, pos[0] - 1)
        row_end = min(self.size, pos[0] + 2)
        col_start = max(0, pos[1] - 1)
        col_end = min(self.size, pos[1] + 2)
        

        # Iterate over the rows
        for i in range(row_start, row_end):
            # Iterate over the columns
            for j in range(col_start, col_end):
                # Add cell to the subgrid
                indices.append(self.grid[i][j])

        '''

        indices = [state]

        

        return indices

    def index2status(self, s):
        return s % self.size ** 2, s // self.size ** 2

    def generate_neighbour_matrix(self):

        self.nbhd_matrix_blue = np.zeros((self.size ** 2, self.n_blue_states))
        self.nbhd_matrix_red = np.zeros((self.size ** 2, self.n_red_states))

        for pos in range(self.size ** 2):
            neighbor_indices = self.neighbourhood(pos)
            for state in range(self.size ** 2, 2 * self.size ** 2):
                nbr_pos, nbr_status = self.index2status(state)
                if nbr_pos in neighbor_indices:
                    self.nbhd_matrix_blue[pos, state] = 1
                    self.nbhd_matrix_red[pos, state] = 1

    def status_transition_matrix_blue(self, mu_t, nu_t): #compute [(s' = 0|p, s, mu, nu), (s' = 1|p, s, mu, nu)] only the alive members contribute for each position

        diff = self.rho * mu_t - (1 - self.rho) * nu_t
        interaction = self.nbhd_matrix_blue @ diff

        interaction_kill =  np.clip(- self.alpha_x*interaction, 0, 1)
        interaction_ressurect =  np.clip(self.beta_x*interaction, 0, 1)

        status_trans_blue = np.zeros((2, 2, self.size ** 2))
        status_trans_blue[0, 0, :] = 1 - interaction_ressurect
        status_trans_blue[1, 0, :] = interaction_ressurect
        status_trans_blue[0, 1, :] = interaction_kill
        status_trans_blue[1, 1, :] = 1 - interaction_kill

        status_trans_matrix_blue = status_trans_blue.reshape((2, -1))   # DoA for each position
        return status_trans_matrix_blue


    def status_transition_matrix_red(self, mu_t, nu_t): #compute [(s' = 0|p, s, mu, nu), (s' = 1|p, s, mu, nu)] only the alive members contribute for each position

        diff = (1 - self.rho) * nu_t - self.rho * mu_t 
        interaction = self.nbhd_matrix_red @ diff

        interaction_kill = np.clip(- self.alpha_y * interaction, 0, 1)
        interaction_ressurect = np.clip(self.beta_y * interaction, 0, 1)

        status_trans_red = np.zeros((2, 2, self.size ** 2))
        status_trans_red[0, 0, :] = 1 - interaction_ressurect
        status_trans_red[1, 0, :] = interaction_ressurect
        status_trans_red[0, 1, :] = interaction_kill
        status_trans_red[1, 1, :] = 1 - interaction_kill

        status_trans_red[0, 1, self.target_indices] = 0
        status_trans_red[1, 1, self.target_indices] = 1 

        status_trans_matrix_red = status_trans_red.reshape((2, -1))   # DoA for each position
        return status_trans_matrix_red


    def generate_position_transition_matrix(self):

        self.position_transition_matrix_blue = np.zeros((self.size ** 2, self.n_blue_states, len(self._action_to_direction)))
        self.position_transition_matrix_red = np.zeros((self.size ** 2, self.n_red_states, len(self._action_to_direction)))

        for state in range(self.size ** 2 * 2):
            pos, status = self.index2status(state)
            if status == 0:
                self.position_transition_matrix_blue[pos, state, :] = 1
                self.position_transition_matrix_red[pos, state, :] = 1

            else:
                for action in range(len(self._action_to_direction)):
                    next_pos = np.clip(self.index2pos(pos) + self._action_to_direction[action], 0, self.size - 1)
                    next_pos = self.pos2index(next_pos)
                    self.position_transition_matrix_blue[next_pos, state, action] = 1
                    self.position_transition_matrix_red[next_pos, state, action] = 1

            if np.isin(pos, self.target_indices):
                self.position_transition_matrix_red[:, state, :] = 0
                self.position_transition_matrix_red[pos, state, :] = 1
        
        self.position_transition_matrix_blue_stay = np.zeros((self.size ** 2, self.n_blue_states, len(self._action_to_direction)))
        self.position_transition_matrix_red_stay = np.zeros((self.size ** 2, self.n_red_states, len(self._action_to_direction)))
        for state in range(self.size ** 2 * 2):
            pos, status = self.index2status(state)
            self.position_transition_matrix_blue_stay[pos, state, :] = 1
            self.position_transition_matrix_red_stay[pos, state, :] = 1


        self.position_transition_matrix_blue_stack = np.stack([self.position_transition_matrix_blue_stay, self.position_transition_matrix_blue])
        self.position_transition_matrix_red_stack = np.stack([self.position_transition_matrix_red_stay, self.position_transition_matrix_red])

    
    def get_transition_blue(self, state, u, mu_t, nu_t):  # for a position p, calculate the transition for an action u under mean fields mu and nu

        pos_transition = self.position_transition_matrix_blue[:, state, u]
        pos_stay = self.position_transition_matrix_blue_stay[:, state, u]
        pos = np.stack([pos_stay, pos_transition])
        status_transition = self.status_transition_matrix_blue(mu_t, nu_t)[:, state]
        transitionProb = np.reshape(pos * status_transition[:, None], (1, -1))
        return transitionProb[0]
    
    def get_transition_red(self, state, v, mu_t, nu_t):  # for a position p, calculate the transition for an action u under mean fields mu and nu

        pos_transition = self.position_transition_matrix_red[:, state, v]
        pos_stay = self.position_transition_matrix_red_stay[:, state, v]
        pos = np.stack([pos_stay, pos_transition])
        status_transition = self.status_transition_matrix_red(mu_t, nu_t)[:, state]
        transitionProb = np.reshape(pos * status_transition[:, None], (1, -1))
        return transitionProb[0]
    
    def ask_oracle_blue(self, mu_t, nu_t, blue_policy):

        blue_policy = blue_policy.reshape(self.n_blue_states, self.n_blue_actions)
        blue_policy = normalize(blue_policy, axis=1, norm='l1')

        status_transition = self.status_transition_matrix_blue(mu_t, nu_t)

        tmp = status_transition[:, None, :, None] * self.position_transition_matrix_blue_stack
        tmp = np.reshape(tmp, (self.n_blue_states, self.n_blue_states, len(self._action_to_direction)))

        F = np.einsum('ijk, jk->ij', tmp, blue_policy)

        return np.transpose(F)

    def ask_oracle_red(self, mu_t, nu_t, red_policy):

        red_policy = red_policy.reshape(self.n_red_states, self.n_red_actions)
        red_policy = normalize(red_policy, axis=1, norm='l1')

        status_transition = self.status_transition_matrix_red(mu_t, nu_t)

        tmp = status_transition[:, None, :, None] * self.position_transition_matrix_red_stack
        tmp = np.reshape(tmp, (self.n_red_states, self.n_red_states, len(self._action_to_direction)))

        G = np.einsum('ijk, jk->ij', tmp, red_policy)

        return np.transpose(G)

    def getEmpiricalMeanField(self, observation):
        mean_field_blue = np.zeros(self.n_blue_states)
        mean_field_red = np.zeros(self.n_red_states)

        # make obs into a dict
        for i in range(self.n_blue_agents+self.n_red_agents):
            if i < self.n_blue_agents:
                loc_blue = observation[i]
                mean_field_blue[loc_blue] += 1
            else:
                loc_red = observation[i]
                mean_field_red[loc_red] += 1

        mean_field_blue = (1 / self.n_blue_agents) * mean_field_blue
        mean_field_red = (1 / self.n_red_agents) * mean_field_red
    
        mean_field_obs = np.hstack((mean_field_blue, mean_field_red))
        return {'blue': mean_field_obs, 'red': mean_field_obs}

    def ask_oracle_reward(self, nu_t: np.ndarray, mu_t: np.ndarray):
        reward_vec = (nu_t[np.array(self.target_indices) + self.n_red_states//2]) * self.rho 
        reward = np.sum(reward_vec) - 10*np.sum(nu_t[:self.n_red_states//2])*(1 - self.rho)
        return {'blue': -reward, 'red': reward}


if __name__ == "__main__":
    
    oracle = mfOracle(n_blue_states=18, n_blue_actions=4, n_blue_agents=3,
                      n_red_states=18, n_red_actions=4, n_red_agents=3, target_pos=np.array([2, 2]))

    n_blue_states = 18
    n_red_states = 18

    mu = np.zeros(n_blue_states)
    mu[7] = 0.1
    mu[9] = 0.1
    mu[10] = 0.2
    mu[11] = 0.1
    mu[12] = 0.2
    mu[15] = 0.1
    mu[17] = 0.2


    nu = np.zeros(n_red_states)
    nu[13] = 0.4
    nu[14] = 0.3
    nu[16] = 0.2
    nu[17] = 0.1

    p = oracle.get_transition_blue(17, 3, mu, nu)
    print(p[0, 14])
    print(p[0, 16])

    p = oracle.get_transition_blue(7, 3, mu, nu)
    print(p[0, 7])
    print(p[0, 6])

    p = oracle.get_transition_red(17, 3, mu, nu)
    print(p)

    print('Done!')