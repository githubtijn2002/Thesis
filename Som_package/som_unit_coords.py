# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:58:57 2024

@author: sdd380
"""

import numpy as np
from som_set import som_set
import math

def som_unit_coords(topol, * args, **kwargs):
    # DESCRIPTION
    #
    # Calculates the map grid coordinates of the units of a SOM based on 
    # the given topology. The coordinates are such that they can be used to
    # position map units in space. In case of 'sheet' shape they can be 
    # (and are) used to measure interunit distances. 
    #
    # NOTE: for 'hexa' lattice, the x-coordinates of every other row are shifted 
    # by +0.5, and the y-coordinates are multiplied by sqrt(0.75). This is done 
    # to make distances of a unit to all its six neighbors equal. It is not 
    # possible to use 'hexa' lattice with higher than 2-dimensional map grids.
    #
    # 'cyl' and 'toroid' shapes: the coordinates are initially determined as 
    # in case of 'sheet' shape, but are then bended around the x- or the 
    # x- and then y-axes to get the desired shape. 
    
    # default values
    
    sTopol = som_set('som_topol', *['lattice'], **{'lattice': 'rect'})
    
    # topol
    if isinstance(topol, dict):
        if topol['type']== 'som_map':
            sTopol= topol['topol']
        elif topol['type']== 'som_topol':
            sTopol = topol
    elif isinstance(topol,tuple):
        for i in range(len(topol)-1):
            if isinstance(topol[i], np.ndarray):
                sTopol['msize']= topol[i]
            elif isinstance(topol[i], str):
                if topol[i] in ['rect', 'hexa']:
                    sTopol['lattice']=topol[i]
                elif topol[i] in ['sheet', 'cyl' ,'toroid']:
                    sTopol['shape']= topol[i]
    else:
        sTopol['msize']= topol
    
    if np.prod(sTopol['msize']) == 0:
        raise ValueError('Map size is 0')
    
    # lattice
    if len(args)>0 and 'lattice' in args and bool(kwargs['lattice']):
        sTopol['lattice']= kwargs['lattice']
     
    # shape
    if len(args)>1 and 'shape' in args and bool(kwargs['shape']):
        sTopol['shape']= kwargs['shape']
    
    ### Action
      
    msize = sTopol['msize']
    lattice = sTopol['lattice']
    shape = sTopol['shape']
    
    # init variables
    if len(msize)==1:
        msize = (msize, 1)
    munits = np.prod(msize)
    mdim = len(msize)
    Coords = np.zeros((munits,mdim))
    
    
    # initial coordinates for each map unit ('rect' lattice, 'sheet' shape)
    k = list(np.concatenate(([1], np.cumprod(msize[:-1]))))
    inds = np.arange(munits)
    Coords = np.zeros((len(inds), mdim), dtype=np.float64)

    for i in range(mdim, 0, -1):
        Coords[:, i-1] = np.floor(inds / k[i-1]).astype(int)
        inds = np.remainder(inds, k[i-1])
    
    # change subscripts to coordinates (move from (ij)-notation to (xy)-notation)
    Coords[:, [0, 1]] = np.fliplr(Coords[:, [0, 1]])
    
    # 'hexa' lattice
    if lattice == 'hexa':
        if mdim >2:
            raise ValueError('You can only use hexa lattice with 1- or 2-dimensional maps.')
        # offset x-coordinates of every other row
        inds_for_row = np.arange(msize[1]) * msize[0]
        for i in range(1, msize[0], 2):
            idx = (i+inds_for_row).astype(int)
            Coords[idx,0] = Coords[idx,0] + 0.5



    if shape == 'sheet':
        if lattice == 'hexa':
            # this correction is made to make distances to all neighboring units equal
            Coords[:,1]= Coords[:,1]* np.sqrt(0.75)
    elif shape == 'cyl':
        # to make cylinder the coordinates must lie in 3D space, at least
        if mdim <3:
            Coords = np.hstack((Coords, np.ones((munits, 1))))
            mdim = 3
        # Bend the coordinates to a circle in the plane formed by x- and 
        # and z-axis. Notice that the angle to which the last coordinates
        # are bended is _not_ 360 degrees, because that would be equal to 
        # the angle of the first coordinates (0 degrees).
        Coords[:,0] = Coords[:,0]/ max(Coords[:,0])
        Coords[:,0] = 2* math.pi * Coords[:,0] * msize[1]/ (msize[1]+1)
        Coords[:,0, 2] = np.vstack((np.cos(Coords[:, 0]), np.sin(Coords[:, 0]))).T
        
    elif shape == 'toroid':
        # NOTE: if lattice is 'hexa', the msize(1) should be even, otherwise 
        # the bending the upper and lower edges of the map do not match 
        # to each other
        if lattice == 'hexa':
            print("warning: Map size along y-coordinate is not even")
        # to make toroid the coordinates must lie in 3D space, at least
        if mdim <3:
            Coords= np.hstack((Coords, np.ones((munits, 1))))
            mdim = 3
        # First bend the coordinates to a circle in the plane formed
        	# by x- and z-axis. Then bend in the plane formed by y- and
        # z-axis. (See also the notes in 'cyl').
        
        Coords[:,0] = Coords[:,0]/ max(Coords[:,0])
        Coords[:,0] = 2* math.pi * Coords[:,0] * msize[1]/ (msize[1]+1)
        Coords[:,0, 2] = np.vstack((np.cos(Coords[:, 0]), np.sin(Coords[:, 0]))).T
        
        Coords[:,1] = Coords[:,1]/ max(Coords[:,1])
        Coords[:,1] = 2* math.pi * Coords[:,1] * msize[0]/ (msize[0]+1)
        Coords[:,2] = Coords[:,2] - min(Coords[:,2]) +1
        Coords[:, [1, 2]] = Coords[:, [2, 2]] * np.column_stack((np.cos(Coords[:, 1]), np.sin(Coords[:, 1])))

    return Coords
        
            

            

    

    