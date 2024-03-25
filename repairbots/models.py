from __future__ import annotations
import numpy as np
from typing import Optional



class Layer():
    
    def __init__(self,x_size:int,y_size:int):
        """Initializes the Layer instance with zeros (no objects)."""
        self.x_size = x_size
        self.y_size = y_size
        self.layer = np.zeros((y_size,x_size),dtype=np.int8)
            
    def empty_cells(self) -> list[tuple]:
        """Return a list of all empty cells expressed as tuple of indices of the global state array."""
        return list(map(tuple,np.transpose(np.where(self.layer==0))))
    
    def find_adjacent(self,arr_filter:np.ndarray) -> np.ndarray:
        """For an array of 0's an 1's, encodes cells adjacent to 1's with 1 and returns all others
        (including the original 1's) as 0's."""
        
        arr = self.layer[arr_filter].copy()
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
        

class Agent:    
    obs_range: int = 2 # ! adjust to 5
    
    
    def __init__(self,xy:tuple[int,int],rank:int):
                
        self.position_y = xy[0]
        self.position_x: xy[1]
        self.rank = rank
        self.status = 1 # 1:AVAILABLE, 2: REPAIRING, 3:SEEKING HELP, 4:RESPONDING        
        self.hole_noted = 0        
        self.req_n_bots: Optional[int] = None
        self.to_hole_x: int = 0
        self.to_hole_y: int = 0
        
    @property
    def position(self):
        return (self.position_y,self.position_x)
    
    def move(self,action:int,hole_layer:HoleLayer,agent_layer:AgentLayer):
        
        if action==0:
            delta = [-1,0]
        elif action==1:
            delta = [1,0]
        elif action==2:
            delta = [0,-1]
        elif action==3:
            delta = [0,1]
        
        target_pos = (self.position_y+delta[0],self.position_x+delta[1])
        # update if within bounds and space is empty
        in_bounds = (target_pos[0] >=0 and target_pos[0] < self.layer.shape[0] and 
            target_pos[1] >=0 and target_pos[1] < self.layer.shape[1])
        space_empty = hole_layer[target_pos]==0 and agent_layer[target_pos]==0
        
        if in_bounds and space_empty:
            self.position_y = target_pos[0]
            self.position_x = target_pos[1]
            
            if self.hole_noted==1: # if hole_noted, then adjust relative position
                old_dist_to = abs(self.to_hole_y) + abs(self.to_hole_x)
                self.to_hole_y += delta
                self.to_hole_x += delta
                new_dist_to = abs(self.to_hole_y) + abs(self.to_hole_x)
                
                if new_dist_to < old_dist_to: # small reward if closer to hole
                    reward = 0.5
                # remove hole_noted if target reached
                if self.position_x==self.to_hole_x and self.position_y==self.to_hole_y:                                
                    self.hole_noted = 0
                    self.status = 1
                # should it be removed after a certain number of turns?        
        
        if self.status==4: # if previously repairing, set status to moving
            self.status = 1
        
        ...
    
    def repair(self):
        
        if self.status==1: # status only changes to 4 if previously available, otherwise retains status
            self.status = 4
        ...
    
    def note_hole(self,req_n_bots:int) -> None:
        """Used on agent action selection "NOTE HOLE" which is action masked to only be available with a hole in view 
        and if the agent had an available status. Tracks current position as (0,0) and all subsequent movements are 
        calculated from this position.
        """                        
        self.status = 3
        self.from_hole = (0,0)
        self.req_n_bots = req_n_bots
        reward = 1 # small reward to take advantage of the skill
        return reward
    
    def share_info(self,agent_layer:AgentLayer,agent_status_layer:AgentStatusLayer) -> None:
        """Used on agent action selection "SHARE INFO" which is action masked to only be available with other agents in
        view. For each of those agents, the sharing agent's obshole_relloc is stored to their obshole_relloc. Each 
        agent shared with decreases the sharers target number of sharing available.
        """        
        # ! action mask at env step
        al_av = agent_layer.agent_view(self.rank,relative=False)
        asl_av = agent_status_layer.agent_view(self.rank)[0]        
        other_agents = al_av[(al_av!=0) & (asl_av!=0)]
        
        for other_agent in other_agents: # should only run while req_n_bots > 0
            while self.req_n_bots > 0:
                other_agent.to_hole = self.from_hole + ... # ! add distance from other agents
                self.req_n_bots -= 1
        
        if self.req_n_bots <= 0:
            self.status = 1
        
        # reward communicating for each agent
        rewards = other_agents
        
        return rewards
    
    def internals():
        # generate agent belief state for obs space

class AgentLayer(Layer):
    
    def __init__(self,x_size:int,y_size:int,agents:dict[int:Agent]):
        self.x_size = x_size
        self.y_size = y_size
        self.agents = agents
        # ? need to create layer?
    
    def update(self):
        # update based on all the agent positions        
        new_layer = np.zeros(shape=(self.y_size,self.x_size),dtype=np.int8)
        for rank,agent in self.agents:
            new_layer[agent.position] = rank
        return new_layer
    
    def agent_view(self,agent_id:int,relative=True) -> np.ndarray:
                
        lv = self.local_view(self.agents[agent_id].position,Agent.obs_range,0,0)            
        
        if relative:
            lv = np.where((lv!=0) & (agent_id > lv),1,lv)
            lv = np.where((lv!=0) & (agent_id < lv),-1,lv)
            
        return lv
        
    
        

        
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
    
    # ? default status when starting?
    
    # ! ideally run and store split by status once each turn and then call local view on it
    def split_by_status(self) -> np.ndarray:
            # splits the status layer into multiple binary layers where each status is represented as 0,1
            sl = self.layer
            n_dim_sl = np.array()
            for scode in range(4):
                n_dim_sl.append(np.where(sl==scode,1,0))
            
            return n_dim_sl
    
    def agent_view(self,agent_id) -> np.ndarray[3,3]:

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
        
        center = self.emtpy_cells()[np.random.choice(len(self.empty_cells()))]
        
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
    
class MapLayer(Layer):
    
    def agent_view(self,position):
        return self.local_view(position,Agent.obs_range,1)
