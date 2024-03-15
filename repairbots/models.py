from __future__ import annotations
import numpy as np
from typing import Optional
import utils

# ? implement a grid class that stores all locations and draws them?
    # problem depending on AEC sequenced actions - won't update until all are complete
# ? action-masking is location-relative; how does this work?

# location-based actions
    # communicate requires other agents in view
        # reduces # to communicate
        

# objects store locations or is there a collection of locations and the objects in those locations?
    # and then grid combines them?
    
    # Location can have ordering dunder methods which can then be used to sort and shape

# global state is a convenience class to simplify the main environment class 
# to distinguish process from the functions used to process transactions

# Location is used for object relative processing
    


# 


class GlobalState:
    
    # store x and y dim
    
    def __init__(self, x_dim:int, y_dim:int):
        self.grid = np.zeros((x_dim,y_dim), dtype=np.int8)
        # ? initialize view range? in Location?

    # ALL_COORDS = [(i, j) for i in range(N) for j in range(N)]

    def empty_cells(self) -> list[tuple]:
        """Return a list of all empty cells expressed as tuple of indices of the global state array."""
        return list(map(tuple,np.transpose(np.nonzero(self.grid==0))))

    def update(self):
        
        pass
    
    def render(self):
        
        pass
    
    # REPAIR
        #
    # COMMUNICATE
        #
    # MOVE
        # take each agent's move's target location and sequentially resolve as success or bump
    # HOLE EXPANSION 
        # for each hole ready to expand, finds its adjacents that also hit the probability for expansions, and append to
        # a single collected list. Find those that overlap with empty_cells, and create a hole in each        
    # RANDOM HOLE CREATION
        # for all empty cells, apply a random number and if it meets the threshold then create a hole
            # ? major hole creation process - is it just a part of random hole? different probability?
        
    
    def local_view(self,center:Location,dist:int):
        
        c= center.as_indices()
        
        return self[min(c[1]+dist,0):min(c[1]-dist,self.y_dim),
             min(c[0]+dist,0):min(c[0]-dist,self.x_dim)]
        

class Location:
    x:int
    y:int
    
    def adjacent(self) -> list[Location]:
    
        pass
    
    
        
    def is_adjacent(self, other_loc:Location) -> bool:
        
        pass
    
    def as_indices(self) -> tuple:
        # convert x,y coordinate system to array indices
        pass


# ? create RelativeLocation class? sub-class of Location?
    # could skip the class and just create an array of np.zeros(2, dtype=np.int32) and calculate movement from there
class RelativeLocation(Location):
    
    pass


        

class HoleLayer:
    loc:Location # ? remove?
    dmg_lvl:int=0
    expand_at:int=8
    expand_prob:float=0.2    
    reduce_by:int=2
    # ! need to prevent duplicates at location
    # ? consider never creating instances        
        # since the global state contains the dmg_lvl of all, just vectorize everything?
        # should we vectorize this to just do it for all
    
    
    def __init__(self):
        
        pass
    
    def add(self):
        
        pass

    # create hole_adj function that returns an array of all the adjacent locations
        
    @classmethod
    def reduce(cls,hole_layer:np.ndarray,agent_layer:np.ndarray) -> np.ndarray:
        # ? vectorize?
            # ? create an action layer to determine which agents are taking the repair action?
        # should this be under the agent methods? or completely independent?            
        # called when repair action is taken
        
        
        # find agents taking the repair action
        # find cells adjacent to repairing agents
        
        
        # self.dmg_lvl -= self.reduce_by
        
        pass
    
    @classmethod
    def progress(cls,hole_layer:np.ndarray,agent_layer:np.ndarray) -> np.ndarray:
        """Takes the existing hole layer and expands to adjacent cells where no agents are present based on the 
        class expand_at probability. Then increments the damage level of all other holes by 1."""        
        
        # expand in cells adjacent to full holes
        # deduct the agent_layer which indicates an agent is occupying a possible expansion spot
        expandable_at = np.clip(utils.find_adjacent(hole_layer==cls.expand_at) - agent_layer,0,1)
        expand_roll = expandable_at * np.random.rand(hole_layer.shape)
        expand_to = np.where(expand_roll <= cls.expand_prob,1,0)
        
        # increment other holes
        inc_lvl = np.where(np.logical_and(hole_layer>0, hole_layer<cls.expand_at),hole_layer+1,hole_layer)
        
        new_hole_layer = expand_to + inc_lvl
        
        return new_hole_layer
    
    def draw(self):
        # ? I think everything is drawn at once
        pass
    
class AgentLayer:
    
    # represents the status of agents on the map
    
    # 1:"MOVING"
    #   - white, last took a move action 
    # 2:"REPAIRING"
    #   - green, last took the repair action
    # 3:"SEEKING HELP"
    #   - orange, has a target and to_n_bots is not None
    # 4:"RESPONDING"
    #   - yellow, has a target and to_n_bots is None
    
    # can only communicate to agents that are moving or repairing
    # need to be processed in reverse order (otherwise a responding agent who is moving is considered available)
    
    def __init__():
        
        pass
    
    def move_agents(self, agent_idx, action):
        
        # update target if it exists
        
        pass
    
    
class Agent:    
    obshole_relloc: Optional[RelativeLocation] = None
    req_n_bots: Optional[int] = None
    target_relloc: Optional[RelativeLocation] = None
    
    def add(self): # ? add multiple at once?
        pass
    
    def move(self):
        
        # if obshole_relloc is not none, then also adjust relative position
        
        pass
    
    def note_hole(self,req_n_bots:int) -> None:
        """Used on agent action selection "NOTE HOLE" which is action masked to only be available with a hole in view.
        Tracks current position as (0,0) and all subsequent movements are calculated from this position.
        """                
        self.obshole_relloc = RelativeLocation(0,0)
        self.req_n_bots = req_n_bots
        
        return None
    
    def share_info(self, bots_in_view:list[Agent]) -> None:
        """Used on agent action selection "SHARE INFO" which is action masked to only be available with other agents in
        view. For each of those agents, the sharing agent's obshole_relloc is stored to their obshole_relloc. Each 
        agent shared with decreases the sharers target number of sharing available.
        """        
        for b in bots_in_view:
            b.target_relloc = self.obshole_relloc
            self.req_n_bots -= 1    
        
        return None
        
        