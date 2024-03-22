import numpy as np

def find_adjacent(arr:np.ndarray) -> np.ndarray:
    """For an array of 0's an 1's, encodes cells adjacent to 1's with 1 and returns all others
    (including the original 1's) as 0's."""

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

# def empty_cells(arr:np.ndarray,0) -> list[tuple]:
#         """Return a list of all empty cells expressed as tuple of indices of the global state array."""
#         return list(map(tuple,np.transpose(np.where(arr==0))))