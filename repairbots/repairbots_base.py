import random
import pygame
import numpy as np
from models import MapLayer, Agent, AgentLayer, HoleLayer, AgentStatusLayer
from gymnasium.spaces import Dict, Box, Discrete
from gymnasium.utils import seeding
from pettingzoo.utils import agent_selector
from typing import Optional,Literal
import models


class RepairBots:
    def __init__(
        self,        
        x_size: int = 100,
        y_size: int = 100,
        max_cycles:int=300,
        game_over: float = 0.25,
        shared_reward: bool = True,
        num_agents: int = 20,
        agent_controller: Optional[RepairBotPolicy] = None, # ? might need to adjust
        render_mode:Optional[Literal["human"]]=None,
        FPS:int=15
        
        
    ):
        self.x_size = x_size
        self.y_size = y_size
        self.max_cycles = max_cycles
        self._seed()                
        
        # REWARD SETUP
        # self.shared_reward = shared_reward # ? might need to change this
        # self.local_ratio = 1.0 - float(self.shared_reward)                        
    
        # SPACES
        obs_space = Dict({
            "agent_view":Box(
                low=-1,
                high=8,
                shape=(self.y_size,self.x_size,7), # map,holes,rel_rank,status(4)
                dtype=np.int8
            ),
            "agent_state": Box(
                low=0,
                high=max(self.x_size,self.y_size,self.num_agents),
                shape=(4,), # hole_noted,req_n_bots,to_hole_x,to_hole_y
                dtype=np.int8
            )
        })
        self.observation_space = [obs_space for _ in range(self.num_agents)]
        act_space = Dict({
            "standard_actions":Discrete(7),
            # 0: UP
            # 1: DOWN
            # 2: LEFT
            # 3: RIGHT
            # 4: REPAIR
            # 5: NOTE
            # 6: SHARE
            "req_n_bots":Discrete(self.num_agents-1)
        })
        self.action_space = [act_space for _ in range(self.num_agents)]
        
        # AGENTS
        self.num_agents = num_agents
        if agent_controller is None:
            RandomPolicy(,self.np_random) # ! TODO
        else:
            self.agent_controller = agent_controller

        # RENDER SETTINGS
        self.render_mode = render_mode
        self.screen = None
        self.pixel_scale = 30 * 25
        self.clock = pygame.time.Clock()
        self.FPS = FPS  # Frames Per Second        
        self.frames = 0
        
        self.reset()
        

    def render(self):
        ...

    def observe(self,agent_id):
        ...
        
    

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None        
    
    def reset(self, seed:int):
        
        map_size = self.x_size, self.y_size        
        self.map_layer = MapLayer(map_size)
        
        self.agent_dict = {}
        # create agents randomly around the map        
        locs = random.sample(self.agent_layer.empty_cells(),self.num_agents)
        # indexing starts at 1 to avoid 0 which represents no agents in the space
        for rank,loc in enumerate(locs):
            self.agent_dict[rank+1]=Agent(loc,rank+1)
        self.agent_layer = AgentLayer(map_size,self.agent_dict).update()
        
        self.hole_layer = models.HoleLayer(self.x_size,self.y_size)
        self.agent_status_layer = models.AgentStatusLayer(self.x_size,self.y_size)
            # ? what are the statuses on the first turn?
        self.latest_reward_state = [0 for _ in range(self.num_agents)]                
        
        # return the first agent's observation
        return self.safely_observe(0)
        
        ...
        
    
    def step(self,agent_id, action, is_first:bool, is_last:bool):
        
        agent = self.agents[agent_id]
        # each action step happens individually, however some of the actions are     
    # REPAIR
        #
    # NOTE
    
    # SHARE
        #
    # MOVE
        # take each agent's move's target location and sequentially resolve as success or bump
    
        # 
    
        if is_first:
            # process all repairs
            # process all communications
            # process all moves
            ...
    
        if is_last: # runs post-action of the last agent to act as the "environment's turn"
            # progress and expand existing holes
            self.hole_layer.progress(self.agent_layer)            
            # chance to add new random holes
            if np.random.rand() <= self.add_prob:
                self.hole_layer.add(self.agent_layer)
        
            # update agentstatuslayer
        
        ...


    def observe(self,agent_id:int) -> np.ndarray:
        
        return np.stack([self.map_layer.agent_view(agent_id),
                  self.hole_layer.agent_view(agent_id),
                  self.agent_layer.agent_view(agent_id), # ! review how these 4 layers get parsed
                  self.agent_status_layer.agent_view(agent_id)],axis=2)
        

    # UTIL METHODS
    def _seed(self, seed=None): # ! to review
        self.np_random, seed_ = seeding.np_random(seed)
        try:
            policies = [self.evader_controller, self.pursuer_controller]
            for policy in policies:
                try:
                    policy.set_rng(self.np_random)
                except AttributeError:
                    pass
        except AttributeError:
            pass

        return [seed_]
    
    @property
    def agents(self):
        return self.pursuers
    
    def get_param_values(self):
        return self.__dict__
    
        # # ? SAMPLE from Connect 4
    # # Key
    # # ----
    # # blank space = 0
    # # agent 0 = 1
    # # agent 1 = 2
    # # An observation is list of lists, where each list represents a row
    # #
    # # array([[0, 1, 1, 2, 0, 1, 0],
    # #        [1, 0, 1, 2, 2, 2, 1],
    # #        [0, 1, 0, 0, 1, 2, 1],
    # #        [1, 0, 2, 0, 1, 1, 0],
    # #        [2, 0, 0, 0, 1, 1, 0],
    # #        [1, 1, 2, 1, 0, 1, 0]], dtype=int8)
    # def observe(self, agent):
    #     board_vals = np.array(self.board).reshape(6, 7)
    #     cur_player = self.possible_agents.index(agent)
    #     opp_player = (cur_player + 1) % 2

    #     cur_p_board = np.equal(board_vals, cur_player + 1)
    #     opp_p_board = np.equal(board_vals, opp_player + 1)

    #     observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
    #     legal_moves = self._legal_moves() if agent == self.agent_selection else []

    #     action_mask = np.zeros(7, "int8")
    #     for i in legal_moves:
    #         action_mask[i] = 1

    #     return {"observation": observation, "action_mask": action_mask}