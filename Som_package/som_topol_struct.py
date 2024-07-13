# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:35:43 2024

@author: sdd380
"""


import numpy as np
from som_set import som_set
import math

# som_topol_struct
def som_topol_struct(*args, **kwargs):
    """
    SOM_TOPOL_STRUCT Default values for SOM topology.

    sTopol = som_topol_struct([[argID,] value, ...])

    Input and output arguments ([]'s are optional):
    [argID,  (string) Default map topology depends on a number of 
    value]  (varies) factors (see below). These are given as a 
                        argument ID - argument value pairs, listed below.

    sT       (dict) The ready topology dictionary.

    Topology dictionary contains values for map size, lattice (default is 'hexa')
    and shape (default is 'sheet'). Map size depends on training data and the
    number of map units. The number of map units depends on number of training
    samples.
    """
    # initialize
    # first line in matlab is with som_set
    
    sTopol = som_set('som_topol', *['lattice', 'shape'], **{'lattice': 'hexa', 'shape': 'sheet'})
    # breakpoint()
    D = []
    dlen = np.nan
    dim = 2
    munits = np.nan
    
    
    # args
    i = 0
    while i <= len(args)-1:
        argok = 1
        if isinstance(args[i], str):
            if args[i] == 'dlen':
                dlen = kwargs[args[i]]
                # i=i+1
            elif args[i] == 'munits':
                munits = kwargs[args[i]]
                sTopol['msize'] = 0
                # i=i+1
            elif args[i] == 'msize':
                sTopol['msize'] = kwargs[args[i]]
                # i=i+1
            elif args[i] == 'lattice':
                sTopol['lattice'] = kwargs[args[i]]
                # i=i+1
            elif args[i] == 'shape':
                sTopol['shape'] = kwargs[args[i]]
                # i=i+1
            elif args[i] == 'data':
                
                if isinstance(args[i], dict):
                    D = args[i]['data']  
                else:
                    D= kwargs[args[i]]
                # i =i+1
                dlen, dim = D.shape
            elif args[i] in ['hexa', 'rect']:
                sTopol['lattice'] = kwargs[args[i]]
            elif args[i] in ['sheet', 'cyl', 'toroid']:
                sTopol['shape'] = kwargs[args[i]]
            elif args[i] in ['som_topol', 'sTopol', 'topol']:
                # i =i+1
                if 'msize' in args[i] and np.prod(args[i]['msize']):
                    sTopol['msize'] = args[i]['msize']
                if 'lattice' in args[i]:
                    sTopol['lattice'] = args[i]['lattice']
                if 'shape' in args[i]:
                    sTopol['shape'] = args[i]['shape']
            else:
                argok = 0
        elif isinstance(args[i], dict) and 'type' in args[i]:
            if args[i]['type'] == 'som_topol':
                if 'msize' in args[i] and np.prod(args[i]['msize']):
                    sTopol['msize'] = args[i]['msize']
                if 'lattice' in args[i]:
                    sTopol['lattice'] = args[i]['lattice']
                if 'shape' in args[i]:
                    sTopol['shape'] = args[i]['shape']
            elif args[i]['type'] == 'som_data':
                D = args[i]['data']
                dlen, dim = D.shape
            else:
                argok = 0
        else:
            argok = 0

        if not argok:
            print(f'(som_topol_struct) Ignoring invalid argument #{i}')
        i += 1
    
    if np.prod(sTopol['msize']) == 0 and bool(sTopol['msize']):
        return sTopol
    
    # otherwise, decide msize
    # first (if necessary) determine the number of map units (munits)
    if np.isnan(munits):
        if not np.isnan(dlen):
            munits = np.ceil(5 * np.sqrt(dlen))
        else:
            munits = 100  # just a convenient value
            
    # then determine the map size (msize)
    # breakpoint()
    if dim == 1:  # 1-D data
        
        sTopol['msize'] = [1, np.ceil(munits)]
    elif len(D)<2:
        sTopol['msize']=[]
        sTopol['msize'].append(round(math.sqrt(munits)))
        sTopol['msize'].append(round(munits / sTopol['msize'][0]))
        print(" sTopol first round done")
    else:
        #determine map size based on eigenvalues
        # initialize xdim/ydim ratio using principal components of the input
        # space; the ratio is the square root of ratio of two largest eigenvalues	
        # autocorrelation matrix
        A = np.full((dim, dim), np.inf)
        for i in range(dim):
            column = D[:, i]
            valid_values = column[np.isfinite(column)]
            mean_val = np.mean(valid_values)
            D[:, i] -= mean_val
        for i in range(dim):
            for j in range(i, dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = np.sum(c) / len(c)
                A[j, i] = A[i, j]
        eigvals, eigvecs = np.linalg.eig(A)

        # Sort the eigenvalues in ascending order
        sorted_indices = np.argsort(eigvals)
        sorted_eigvals = eigvals[sorted_indices]

        # Return the sorted eigenvalues
        eigvals = sorted_eigvals   
        
        if eigvals[-1] == 0 or eigvals[-2] * munits < eigvals[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigvals[-1] / eigvals[-2])

        if sTopol['lattice'] == 'hexa':
            # breakpoint()
            sTopol['msize']=[None, None]
            sTopol['msize'][1] = min(munits, round(np.sqrt(munits / ratio * np.sqrt(0.75))))
        else:
            sTopol['msize'][1] = min(munits, round(np.sqrt(munits / ratio)))
        sTopol['msize'][0]= round(munits/sTopol['msize'][1]) 
        if min(sTopol['msize'])==1:
            sTopol['msize']=(1, max(sTopol['msize']))
            
    return sTopol