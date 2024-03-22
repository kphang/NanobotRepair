from __future__ import annotations
import numpy as np
import random
from typing import Optional

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

class Layer():
    
    def __init__(self,x_size:int,y_size:int):
        """Initializes the Layer instance with zeros (no objects)."""
        self.layer = np.zeros((y_size,x_size),dtype=np.int8)
            
    def empty_cells(self) -> list[tuple]:
        """Return a list of all empty cells expressed as tuple of indices of the global state array."""
        return list(map(tuple,np.transpose(np.where(self.layer==0))))
    
    def find_adjacent(self,filter:np.array) -> np.ndarray:
        """For an array of 0's an 1's, encodes cells adjacent to 1's with 1 and returns all others
        (including the original 1's) as 0's."""
        
        arr = self.layer[filter].copy()
        if (arr>1).any() or (arr<0).any():        
            raise ValueError("arr must contain only 0's or 1's")
            
        y_len, x_len = arr.shape
        
        # shifts in each direction so that 1's will be put in all adjacent cells to the original
        l = np.c_[arr[:,1:],np.zeros(y_len)] 
        r = np.c_[np.zeros(y_len),arr[:,:-1]]
        u = np.r_[arr[1:,:],[np.zeros(x_len)]]
        d = np.r_[[np.zeros(x_len)],arr[:-1,:]]
            
        new_arr = np.clip((l+r+u+d) - arr,0,1) # deduct the original, clip negatives

        return new_arr
        
    def local_view(self,center:tuple[int,int],dist:int,pad_val:int|None,
                   replace_center:Optional[int]=None,ret_sliceidx:bool=False
                   ) -> np.ndarray|tuple[np.ndarray]:
        """Creates a sub-array of the layer centered around a point with a given number of rows and columns surrounding.
        If the center and distance would result in a shape outside the bounds of the original layer, the available data
        is extracted and padded."""
        
        arr = self.layer.copy()
        if replace_center:
            arr[center] = replace_center

        top = center[0]-dist
        bottom = center[0]+dist+1
        left = center[1]-dist
        right = center[1]+dist+1
        
        try:
            sliceidx = np.s_[top:bottom,left:right]
            local_view = arr[sliceidx]
            assert local_view.shape == (2*dist+1,2*dist+1)
            
        except AssertionError:
            sliceidx = np.s_[max(top,0):min(bottom,arr.shape[0]),
                        max(left,0):min(right,arr.shape[1])]
            layer_area = arr[sliceidx]
            
            if pad_val is None:
                local_view = layer_area
            else:
                top_pad = abs(top) if top <0 else 0
                bottom_pad = bottom-arr.shape[0] if bottom >arr.shape[0] else 0
                left_pad = abs(left) if left <0 else 0
                right_pad = right-arr.shape[1] if right >arr.shape[1] else 0            

                local_view = np.pad(layer_area,((top_pad,bottom_pad),(left_pad,right_pad)),"constant",constant_values=pad_val)        
            
        if ret_sliceidx:
            return local_view, sliceidx
        else:
            return local_view
        
        # ? add an arg to replace the center with another value like 0

class Agent:    
    obs_range: int = 2 # ! adjust to 5
    
    
    def __init__(self,xy:tuple[int,int],rank:int):
        
        # test xy in bounds
        self.position = xy
        self.rank = rank
        from_hole: Optional[tuple[int,int]] = None
        req_n_bots: Optional[int] = None
        to_hole: Optional[tuple[int,int]] = None
        
        
    
    def move(self):
        
        # if obshole_relloc is not none, then also adjust relative position
        
        ...
    
    def note_hole(self,req_n_bots:int) -> None:
        """Used on agent action selection "NOTE HOLE" which is action masked to only be available with a hole in view.
        Tracks current position as (0,0) and all subsequent movements are calculated from this position.
        """                
        self.obshole_relloc = (0,0)
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

class AgentLayer(Layer):
    
    def __init__(self,x_size:int,y_size:int, n_agents):
        # agents are identified by a number which also represents their rank
        # actions are processed sequentially according to rank to minimize co-ordination problems
        super().__init__(x_size,y_size)
        self.agents = {}
        
        locs = random.sample(self.empty_cells(),n_agents)
        # indexing starts at 1 to avoid 0 which represents no agents in the space
        for rank,loc in enumerate(locs):
            self.agents[rank+1]=Agent(loc,rank+1)
            self.layer[loc] = rank+1
        
            
    def agent_view(self,agent_idx) -> np.ndarray:
        
        lv = self.local_view(self.agents[agent_idx].position,Agent.obs_range,0,0)            
        lv2 = np.where((lv!=0) & (agent_idx > lv),1,lv)
        lv3 = np.where((lv2!=0) & (agent_idx < lv2),-1,lv2)
        return lv3
        
    
    def process_move(self, agent_idx, action, hole_layer):
        
        # moves are processed sequentially by agent_idx
        
        
        #[[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]
        # 
        
        # update target if it exists
        
        ...
    
    def process_repairs(self, agent_idx, action):
        
        ...    
    
    
                            
    

        
class AgentStatusLayer(Layer):
    
    # represents the status of agents on the map
    
    # 1:"NO STATUS"
    #   - white, last took a move action 
    # 2:"REPAIRING"
    #   - green, last took the repair action
    # 3:"SEEKING HELP"
    #   - orange, has a target and to_n_bots is not None
    # 4:"RESPONDING"
    #   - yellow, has a target and to_n_bots is None
    
    # can only communicate to agents that are moving or repairing
    # need to be processed in reverse order (otherwise a responding agent who is moving is considered available)
    # is the status layer only to help with rendering and action masking?
    # can it help the agents make decisions? if not then don't need to give them access as an observation
    
    # ! ideally run and store split by status once each turn and then call local view on it
    def split_by_status(self) -> np.ndarray:
            # splits the status layer into multiple binary layers where each status is represented as 0,1
            sl = self.layer
            n_dim_sl = np.array()
            for scode in range(4):
                n_dim_sl.append(np.where(sl==scode,1,0))
            
            return n_dim_sl
    
    def agent_view(self,agent_idx) -> np.ndarray[3,3]:

        ...


class HoleLayer(Layer):     
    add_prob:float=0.05
    expand_at:int=8
    progress_prob:float=0.2
    reduce_by:int=2    
    
    # FUTURE: generalize the addition of other layers to take into account if map layer has obstacles
    
    def add(self,agent_layer:AgentLayer) -> None:
        """Spawns holes in a random location of the layer. The approximate size of the hole will be 
        5x5, however may be cut off if spawned in corners. To actual size and formation is not deterministic.
        The center will be a single level 8 hole, surrounded by holes ranging from 5-8, which are then surrounded
        by cells ranging from 0 (no-hole) to 4."""        
        
        center = random.choice(self.empty_cells())        
        
        hole_area1, hole_area_idx1 = self.local_view(center,dist=1,pad_val=None,ret_sliceidx=True)        
        hole_area2, hole_area_idx2 = self.local_view(center,dist=2,pad_val=None,ret_sliceidx=True)
        
        # apply holes from outside in
        self.layer[hole_area_idx2] = np.where((hole_area2==0) & (agent_layer.layer[hole_area_idx2]==0),
                    np.random.choice([0,1,2,3,4],size=hole_area2.shape,p=[.4,.25,.15,.10,.10]),
                    hole_area2
                    )
        print(self.layer)
        self.layer[hole_area_idx1] = np.where((hole_area1==0) & (agent_layer.layer[hole_area_idx1]==0),
                    np.random.choice([5,6,7,8],size=hole_area1.shape,p=[.4,.3,.2,.1]),
                    hole_area1
                    )
        print(self.layer)
        self.layer[center] = 8
        

            
    def reduce(self,agent_layer:AgentLayer) -> None:
        # ? vectorize?
            # ? create an action layer to determine which agents are taking the repair action?
        # should this be under the agent methods? or completely independent?            
        # called when repair action is taken
        
        
        # find agents taking the repair action
        # find cells adjacent to repairing agents
        
        
        # self.dmg_lvl -= self.reduce_by
        
        #return None
        pass
        
    def progress(self,agent_layer:AgentLayer) -> None:
        """Takes the existing hole layer and expands to adjacent cells where no agents are present based on
        progress_prob. For other holes, if they are adjacent to a full hole, then their damage increments by 1.
        If they are not adjacent, then they increase by 1 depending on the progress_prob."""
        
        full_hole_adj = self.find_adjacent(self.layer==self.expand_at)
        other_holes = np.logical_and(self.layer>0, self.layer<self.expand_at)
        
        # expand in cells adjacent to full holes excluding agent_layer and other_holes which indicate blocking objects
        expandable_at = np.clip(full_hole_adj - agent_layer - other_holes,0,1)        
        expand_to = np.where(expandable_at * np.random.rand(self.layer.shape) >= 1-self.progress_prob,1,0)        
        
        # increment other holes adj to full holes
        adj_inc = np.where(np.logical_and(full_hole_adj,other_holes),1,0)        
        
        # probabilistically increment other holes not adj to full holes
        other_nonadj_holes = np.clip(other_holes-full_hole_adj,0,1)
        rand_inc = np.where(other_nonadj_holes * np.random.rand(self.layer.shape) >= 1-self.progress_prob,1,0)        
        
        new_hole_layer = self.layer + expand_to + adj_inc + rand_inc
        
        self.layer = new_hole_layer
        return None
    
    # TODO: draw function
    
