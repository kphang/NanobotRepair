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
        
    
    def local_view(self,center:tuple[int,int],dist:int):
        
        c= center.as_indices()
        
        return self[min(c[1]+dist,0):min(c[1]-dist,self.y_size),
             min(c[0]+dist,0):min(c[0]-dist,self.x_size)]
        

        

class HoleLayer:     
    add_prob:float=0.05
    expand_at:int=8
    progress_prob:float=0.2
    reduce_by:int=2
    # ! need to prevent duplicates at location
    # ? consider never creating instances        
        # since the global state contains the dmg_lvl of all, just vectorize everything?
        # should we vectorize this to just do it for all
    
    
    def __init__(self,x_size:int,y_size:int):
        """Initializes the HoleLayer instance with no holes."""
        self.layer = np.zeros((y_size,x_size),dtype=np.int8)
        
    
    def add(self):
        """Spawns random holes of varying sizes and densities."""
        #np.random.choice([0,1],size=(y_size,x_size),p=[0.95,0.05])   
                
        # occurs with a 5% chance (captured in the environment or the add function itself?)
        # On occurence, select a random unoccupied cell as the centre which is 8
        # For a 3x3 matrix around it, any non-occupied cell is assigned by:
            # np.random.choice([5,6,7,8],size=(3,3),p=[40,30,20,10])
        # For a 5x5 matrix around it, any non-occupied cell is assigned by:
            # np.random.choice([0,1,2,3,4],size=(5,5),p=[40,25,15,10,10])
             
        pass

    # create hole_adj function that returns an array of all the adjacent locations
            
    def reduce(self,hole_layer:np.ndarray,agent_layer:np.ndarray) -> None:
        # ? vectorize?
            # ? create an action layer to determine which agents are taking the repair action?
        # should this be under the agent methods? or completely independent?            
        # called when repair action is taken
        
        
        # find agents taking the repair action
        # find cells adjacent to repairing agents
        
        
        # self.dmg_lvl -= self.reduce_by
        
        return None
        
    def progress(self,hole_layer:np.ndarray,agent_layer:np.ndarray) -> None:
        """Takes the existing hole layer and expands to adjacent cells where no agents are present based on
        progress_prob. For other holes, if they are adjacent to a full hole, then their damage increments by 1.
        If they are not adjacent, then they increase by 1 depending on the progress_prob."""        
        
        full_hole_adj = utils.find_adjacent(hole_layer==self.expand_at)
        other_holes = np.logical_and(hole_layer>0, hole_layer<self.expand_at)        
        
        # expand in cells adjacent to full holes excluding agent_layer and other_holes which indicate blocking objects
        expandable_at = np.clip(full_hole_adj - agent_layer - other_holes,0,1)        
        expand_to = np.where(expandable_at * np.random.rand(hole_layer.shape) >= 1-self.progress_prob,1,0)        
        
        # increment other holes adj to full holes
        adj_inc = np.where(np.logical_and(full_hole_adj,other_holes),1,0)        
        
        # probabilistically increment other holes not adj to full holes
        other_nonadj_holes = np.clip(other_holes-full_hole_adj,0,1)
        rand_inc = np.where(other_nonadj_holes * np.random.rand(hole_layer.shape) >= 1-self.progress_prob,1,0)        
        
        new_hole_layer = hole_layer + expand_to + adj_inc + rand_inc
        
        self.layer = new_hole_layer
        return None
    
    # TODO: draw function
    
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
        
        