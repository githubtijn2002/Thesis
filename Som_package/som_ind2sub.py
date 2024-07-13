# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:31:11 2024

@author: sdd380
"""
import numpy as np

def som_ind2sub(msize, inds):
    
    if isinstance(msize, dict):
        if msize['type'] == 'som_map':
            msize = msize['topol']['msize']
        elif msize['type'] == 'som_topol':
            msize = msize['msize']
        else:
            raise ValueError('Invalid first argument')
        
    n = len(msize)
    k = np.insert(np.cumprod(msize[:-1]), 0, 1)
    # Initialize Subs array
    Subs = np.zeros((len(inds), n), dtype=int)

    # Loop in reverse order
    for i in range(n-1, -1, -1):
        Subs[:, i] = np.floor_divide(inds, k[i])
        
        inds = np.mod(inds, k[i])
    
    return Subs
