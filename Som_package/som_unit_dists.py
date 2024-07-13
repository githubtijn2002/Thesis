# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:13:45 2024

@author: sdd380
"""
import numpy as np
from som_set import som_set
import inspect
from som_neigborhood import som_neigborhood
from som_unit_coords import som_unit_coords
import math

def som_unit_dists(topol, *args, **kwargs):
    
    # check arguments
    if topol is None:
       raise ValueError("The mandatory argument 'topol' is missing.")
   
   # Count the number of arguments
    num_args = 1 + len(args) + len(kwargs)
   
    if num_args < 1:
        raise ValueError(f"Too few arguments: expected at least 2, got {num_args}")
    if num_args > 3:
        raise ValueError(f"Too many arguments: expected at most inf, got {num_args}")
    
    # topol
    sTopol = som_set('som_topol', *['lattice'], **{'lattice': 'rect'})
    
    if isinstance(topol, dict):
        if topol['type']=='som_map':
            sTopol= topol['topol']
        elif topol['type'] == 'som_topol':
            sTopol = topol
    elif isinstance(topol, list):
        for i in range(topol):
           if isinstance(topol[i], np.ndarray):
               sTopol['msize']= topol[i]
           elif isinstance(topol[i], str):
               if topol[i] in {'rect', 'hexa'}:
                   sTopol['lattice']= topol[i]
               elif topol[i] in { 'sheet', 'cyl', 'toroid'}:
                   sTopol['shape'] = topol[i]
    else:
        sTopol['msize'] = topol
    if np.prod(sTopol['msize']) ==0:
        raise ValueError('Map size is 0')
    
    # lattice
    if num_args > 1 and bool(lattice) and not np.isnan(lattice):
        sTopol['lattice']= lattice
    # shape
    if num_args >2 and bool(shape) and not np.isnan(shape):
        sTopol['shape']= shape
    
    #### Action
    msize = sTopol['msize']
    lattice = sTopol['lattice']
    shape = sTopol['shape']
    
    munits = np.prod(msize)
    Ud = np.zeros((munits,munits))
    
    # free topology
    if lattice == 'free':
        N1 = sTopol['connection']
        Ud = som_neigborhood(N1, Inf)
    
    # coordinates of map units when the grid is spread on a plane
    Coords = som_unit_coords(msize, *['lattice', 'shape'], **{'lattice': lattice, 'shape': shape})
    
    # width and height of the grid
    dx = max(Coords[:,0]) - min(Coords[:,0]) 
    if msize[0]>1:
        dx = dx * msize[0] / (msize[0]-1)
    else:
        dx = dx+1
    dy = max(Coords[:,1]) - min(Coords[:,1])
    if msize[1]>1:
        dy = dy * msize[1] / (msize[1]-1)
    else:
        dy = dy+1
    
    # calculate distance from each location to each other location
    if shape == 'sheet':
        for i in range(0,munits-1):
            inds = list(range(i + 1, munits))
            ones_vec = np.ones((munits-1 - i, 1)) * i
            Dco = (Coords[inds, :] - Coords[ones_vec.flatten().astype(int), :]).T
            Ud[i,inds] = np.sqrt(np.sum(Dco**2, axis=0))
            
    elif shape == 'cyl':
        for i in range(0,munits-1):
            inds = list(range(i + 1, munits))
            print(inds)
            ones_vec = np.ones((munits-1 - i, 1)) * i
            Dco = (Coords[inds, :] - Coords[ones_vec.flatten().astype(int), :]).T
            dist = np.sum(Dco**2)
            # The cylinder shape is taken into account by adding and substracting
            # the width of the map (dx) from the x-coordinate (ie. shifting the
            # map right and left).
            DcoS = Dco.copy()
            DcoS[0,:] = DcoS[0,:] +dx
            dist = min((dist, np.sum(DcoS**2)))
            DcoS = Dco.copy()
            DcoS[0,:] = DcoS[0,:] -dx
            dist = min((dist, np.sum(DcoS**2)))
            Ud[i,inds] = math.sqrt(dist)
            
    elif shape == 'toroid':
         for i in range(0, munits-1):
            inds = list(range(i + 1, munits))
            # to be finished    
    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    Ud = Ud + Ud.T
    return Ud