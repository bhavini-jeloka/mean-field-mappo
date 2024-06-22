import functools

import gymnasium
import numpy as np
from gymnasium import spaces, Env
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

import pygame
import random
import copy
from .mfOracle import mfOracle

size = 8
target = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
def BattleFieldEnv(render_mode=None):

    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = BattleField(render_mode=internal_render_mode)

    # this wrapper helps error handling for discrete action spaces
    #env = wrappers.AssertOutOfBoundsWrapper(env)

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    #env = wrappers.OrderEnforcingWrapper(env)

    return env


class BattleField(Env):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "gridworld_v1", "render_fps": 4}

    def __init__(self, render_mode=None, size=8):

        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.numAgents = 20 #8
        self.numActions = 5
        self.num_teams = 2
        self.target = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

        self.possible_agents_blue = ["blue1", "blue2", "blue3", "blue4", "blue5", "blue6", "blue7", "blue8", "blue9", "blue10", 
                                     "blue11", "blue12", "blue13", "blue14", "blue15", "blue16", "blue17", "blue18", "blue19", "blue20"]
        self.possible_agents_red = ["red1", "red2", "red3", "red4", "red5", "red6", "red7", "red8", "red9", "red10", 
                                    "red11", "red12", "red13", "red14", "red15", "red16", "red17", "red18", "red19", "red20"]   


        #self.possible_agents_blue = ["blue1", "blue2", "blue3", "blue4", "blue5", "blue6", "blue7", "blue8"]
        #self.possible_agents_red = ["red1", "red2", "red3", "red4", "red5", "red6", "red7", "red8"]

        # a mapping between agent name and ID
        self.agent_name_mapping_blue = dict(
            zip(self.possible_agents_blue, list(range(len(self.possible_agents_blue))))
        )

        self.agent_name_mapping_red = dict(
            zip(self.possible_agents_red, list(range(len(self.possible_agents_red))))
        )

        self.agent_name_mapping = dict(
            zip(self.possible_agents_blue+self.possible_agents_red, list(range(len(self.possible_agents_blue)+len(self.possible_agents_red))))
        )

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """

        self.window = None
        self.clock = None

        self.mf_oracle = mfOracle(2 * self.size ** 2, self.numActions, 2 * self.size ** 2, self.numActions, self.numAgents,
                                  self.numAgents, self.target)
        self.border_indices = self.find_border_locations()

        # configure spaces
        self.action_space = {0: [], 1: []}
        self.observation_space = {0: [], 1: []}
        self.share_observation_space = {0: [], 1: []}
        share_obs_dim = 0

        self.team_mapping = {'blue': 0, 'red': 1}

        for team_id in range(self.num_teams):
            for agent in range(self.numAgents):
                self.action_space[team_id].append(Discrete(5))
                
                # observation space
                obs_dim = 2*2*size**2
                share_obs_dim += 1
                self.observation_space[team_id].append(spaces.Box(0, obs_dim - 1, shape=(obs_dim + 1,), dtype=np.float32))  # [-inf,inf]
            share_obs_dim = obs_dim
            self.share_observation_space[team_id] = [spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.numAgents)]

    def render(self):
        # Renders the environment. In human mode, it opens up a graphical window that a human can see and understand.
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # We draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * target,
                (pix_square_size, pix_square_size),
            ),
        )

        # First we draw the blue agents
        for i in range(self.numAgents):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self.state[self.agents_blue[i]] + 0.25) * pix_square_size,
                pix_square_size / 6,
            )
            

        # Now we draw the red agents
        for j in range(self.numAgents):
            pygame.draw.circle(
                canvas,
                (255, 0, 0),  
                (self.state[self.agents_red[j]] + 0.75) * pix_square_size,
                pix_square_size / 6  
            )

       
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        return self.observations

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def get_state(self, agent):
        return self.local_state[agent]

    def find_border_locations(self):
        n = self.size
        border_locations = []

        # Top border
        for j in range(n):
            border_locations.append(np.array([0, j]))

        # Bottom border
        for j in range(n):
            border_locations.append(np.array([n - 1, j]))

        # Left border (excluding corners, as they're already covered)
        for i in range(1, n - 1):
            border_locations.append(np.array([i, 0]))

        # Right border (excluding corners)
        for i in range(1, n - 1):
            border_locations.append(np.array([i, n - 1]))

        border_indices = []    

        for border in border_locations:
            border_indices.append(self.mf_oracle.pos2index(border) + self.size ** 2) 

        return border_indices
    
    
    def set_seed(self, seed=None):
        if seed is None:
            self.seed_value = 1
        else:
            self.seed_value = seed


    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """

        random.seed(self.seed_value)

        self.agents_blue = self.possible_agents_blue[:]
        self.agents_red = self.possible_agents_red[:]
        self.rewards_blue = {agent: 0 for agent in self.agents_blue}
        self.rewards_red = {agent: 0 for agent in self.agents_red}
        self._cumulative_rewards_blue = {agent: 0 for agent in self.agents_blue}
        self._cumulative_rewards_red = {agent: 0 for agent in self.agents_red}

        self.agents = self.possible_agents_blue[:] + self.possible_agents_red[:]
        self.rewards = self.rewards_blue | self.rewards_red
        self._cumulative_rewards = self._cumulative_rewards_blue | self._cumulative_rewards_red
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.local_state = {agent: np.random.choice(np.arange(self.size ** 2, 2*self.size**2)) for agent in self.agents} # each row represents an agent's position and status
    
        self.local_states_all = np.array([arr for arr in self.local_state.values()])
        mf = self.mf_oracle.getEmpiricalMeanField(self.local_states_all)
        self.observations = {agent: np.hstack((mf['blue'], self.local_state[agent])) for agent in self.agents}

        """
        Our agent_selector utility allows easy cyclic stepping through the agents list. # keeping these to make it compatible with petting zoo
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observations


    def get_new_state(self, state, action, agent, oracle, size, mf):

        numStates = 2*size**2

        if agent.find('red') != -1:
            idx_prob_dist = oracle.get_transition_red(state, action, mf['blue'][0:numStates], mf['red'][numStates:2*numStates])
            dirIdx = np.random.choice(np.arange(numStates), p=idx_prob_dist)  
        else:
            idx_prob_dist = oracle.get_transition_blue(state, action, mf['blue'][0:numStates], mf['red'][numStates:2*numStates])
            dirIdx = np.random.choice(np.arange(numStates), p=idx_prob_dist)  

        return dirIdx


# TODO: fix observations
    def step(self, jointAction):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """

        numStates = 2*self.size**2

        # new state
        mf = self.mf_oracle.getEmpiricalMeanField(self.local_states_all)

        for agent, action in zip(self.agents, jointAction):
            self.local_state[agent] = self.get_new_state(self.local_state[agent], action, agent, self.mf_oracle, size, mf)

        self.local_states_all = np.array([arr for arr in self.local_state.values()])
        mf = self.mf_oracle.getEmpiricalMeanField(self.local_states_all)

        self.observations = {agent: np.hstack((mf['blue'], self.local_state[agent])) for agent in self.agents}
        reward = self.mf_oracle.ask_oracle_reward(mf['red'][numStates:2*numStates], mf['blue'][0:numStates])

        self.rewards_blue = {agent: reward['blue'] for agent in self.agents_blue}
        self.rewards_red = {agent: reward['red'] for agent in self.agents_red}
        self.rewards = self.rewards_blue | self.rewards_red

        pos_list, status_list = self.mf_oracle.index2status(self.local_states_all)

        dones = {}
        dones[0] = np.logical_not(status_list[:self.numAgents].astype(bool))
        dones[1] = np.logical_not(status_list[:self.numAgents].astype(bool))

        if self.render_mode == "human":
            self._render_frame()

        return self.observations, self.rewards, dones, self.infos
