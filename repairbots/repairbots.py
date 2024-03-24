"""
This module sets up the generic framework for PettingZoo to standardize access for RL.
This is set up as an AECEnv but meant to be run as an ParallelEnv (accomplished through the wrapper).
This was done to help ensure order is enforced.
"""

from gymnasium.utils import EzPickle
from gymnasium.spaces import Space
from pettingzoo import AECEnv,ParallelEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
import pygame
from typing import Optional,Literal,TypeVar
from repairbots_base import FPS
from repairbots_base import RepairBots as _env

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.ClipOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"], # ? remove rgb array?
        "name": "repairbots_v0",
        "is_parallelizable": True,
        "render_fps": 15,
        "has_manual_policy":False
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)        
        self.env = _env(*args, **kwargs)
        self.render_mode = kwargs.get("render_mode")
        if self.render_mode=="human":
            pygame.init()
        
        # initialize standard attributes required by framework
        self.agents = [a+1 for a in range(self.env.num_agents)] # name also indicates rank starting at 1
            # since the ranked agents are represented in observations and 0's indicate no agents we need to start at 1
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
                
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.action_spaces = dict(zip(self.agents, self.env.action_space))        

        self.render_mode = self.env.render_mode
        self.closed = False
    
    def reset(self, seed:Optional[int]=None, options:Optional[dict]=None) -> tuple[dict[AgentID, ObsType],dict[AgentID,dict]]:        
        """
        Resets the environment and returns a dictionary of observations to start the game.
        """
        if seed is not None:
            self.env._seed(seed=seed)
        
        self.steps = 0
        self.agents = self.possible_agents[:]
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.env.reset()


    def render(self):
        if not self.closed:
            return self.env.render()
    
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        self.env.step(
            action, self.agent_name_mapping[agent], self._agent_selector.is_first(), self._agent_selector.is_last()
        )
        
        for k in self.terminations:
            if self.env.frames >= self.env.max_cycles:
                self.truncations[k] = True
            else:
                self.terminations[k] = self.env.is_terminal
        
        if self._agent_selector.is_last():
            for k in self.agents:
                self.rewards[k] = self.env.latest_reward_state[self.agent_name_mapping[k]]
            self._cumulative_rewards[self.agent_selection] = 0 # ? why is it 0?
        
            self.steps += 1 # ? why is steps incremented at each agent?
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()

            if self.render_mode == "human":
                self.render()

    
    
    def close(self):
        """Close API that calls the close function of the base environment used"""
        if not self.closed:
            self.closed = True
            self.env.close()
                    
        # if self.has_reset:
        #     self.env.close()
            
    def state(self):
        """
        Returns the state. State returns a global view of the environment.
        """
        # ! TODO
        ...
    
    def observation_space(self, agent) -> Space:
        """Return the observation space of a given agent."""                            
        return self.observation_spaces[agent]

    def action_space(self, agent) -> Space:
        """Return the action space of a given agent."""
        return self.action_spaces[agent]        

    def observe(self, agent):
        """Allows viewing the specified agent's observation"""
        return self.env.observe(self.agent_name_mapping[agent])


