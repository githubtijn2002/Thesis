# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:55:55 2024

@author: sdd380
"""
from som_set import som_set
import numpy as np
from som_unit_dists import som_unit_dists
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix


def som_unit_neighs(topol, *args):
    
    # PURPOSE
    #
    # Find the adjacent (in 1-neighborhood) units for each map unit of a SOM
    # based on given topology.
    #
    # SYNTAX
    #
    #  Ne1 = som_unit_neighs(sMap);
    #  Ne1 = som_unit_neighs(sM.topol);
    #  Ne1 = som_unit_neighs(msize);
    #  Ne1 = som_unit_neighs(msize,'hexa');
    #  Ne1 = som_unit_neighs(msize,'rect','toroid');
    #
    # DESCRIPTION
    #
    # For each map unit, find the units the distance of which from 
    # the map unit is equal to 1. The distances are calculated
    # along the map grid. Consider, for example, the case of a 4x3 map. 
    # The unit ('1' to 'C') positions for 'rect' and 'hexa' lattice (and
    # 'sheet' shape) are depicted below: 
    # 
    #   'rect' lattice           'hexa' lattice
    #   --------------           --------------
    #      1  5  9                  1  5  9
    #      2  6  a                   2  6  a
    #      3  7  b                  3  7  b
    #      4  8  c                   4  8  c
    #
    # The units in 1-neighborhood (adjacent units) for unit '6' are '2','5','7'
    # and 'a' in the 'rect' case and '5','2','7','9','a' and 'b' in the 'hexa'
    # case. The function returns a sparse matrix having value 1 for these units.  
    # Notice that not all units have equal number of neighbors. Unit '1' has only 
    # units '2' and '5' in its 1-neighborhood. 
    
    # default values
    sTopol = som_set('som_topol', *['lattice'], **{'lattice': 'rect'})
    
    if isinstance(topol, dict):
        if topol['type']== 'som_map':
            sTopol = topol['topol']
        elif topol['type'] == 'som_topol':
            sTopol = topol
    elif isinstance(topol, list):
        for i in range(len(topol)):
            if isinstance(topol[i], np.ndarray):
                sTopol['msize'] = topol[i]
            elif isinstance(topol[i], str):
                if topol[i] in {'hexa', 'rect'}:
                    sTopol['lattice']= topol[i]
                elif topol[i] in {'sheet', 'cyl', 'toroid'}:
                    sTopol['shape']= topol[i]
    else:
        sTopol['msize']= topol
        
    if np.prod(sTopol['msize'])==0:
        raise ValueError('Map size is 0.')
    
    
    # lattice
    if len(args)>0 and bool(lattice) and not np.isnan(lattice):
        sTopol['lattice']= lattice
    
    # shape
    if len(args)>1 and bool(shape) and not np.isnan(shape):
        sTopol['shape'] = shape
    
    ### ACTION
    Ud = som_unit_dists(sTopol)
    
    # 1-neighborhood are those units the distance of which is equal to 1
    munits = np.prod(sTopol['msize'])
    Ne1 = coo_matrix((munits,munits))
    rows = []
    cols = []
    data = []
    for i in range(munits):
        inds = np.where((Ud[i,:]< 1.01) & (Ud[i,:]>0))[0]
        rows.extend([i] * len(inds))  # Repeat i len(inds) times
        cols.extend(inds)
        data.extend(np.ones(len(inds)))  # Add 1s for each valid index
    Ne1 = coo_matrix((data, (rows, cols)), shape=(munits, munits))
    Ne1.data[:]=1
  
    return Ne1 
    