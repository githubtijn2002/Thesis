# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:18:03 2024

@author: sdd380
"""

import numpy as np
from som_set import som_set
from datetime import datetime

def som_map_struct(dim,*args, **kwargs):
    # default values
    
    
    
   
    sTopol = som_set('som_topol', *['lattice', 'shape'], **{'lattice': 'hexa', 'shape': 'sheet'})
    neigh = 'gaussian'
    mask = np.ones((dim, 1))
    name = f'SOM {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'  # Update datetime formatting
    labels = [''] * np.prod(sTopol['msize'])
    comp_names = [f'Variable{i}' for i in range(1, dim + 1)]
    comp_norm = [None] * dim
    
    # args
    # if args:
    i = 0 
    
    while i <= len(args)-1:
        argok = 1
        if isinstance(args[i], str):
            if args[i] == 'mask':
                mask = kwargs[args[i]]
                
            elif args[i] == 'msize':
                sTopol['msize'] = kwargs[args[i]]
                
            elif args[i] == 'labels':
                labels = kwargs[args[i]]
                
            elif args[i] == 'name':
                name = kwargs[args[i]]
                
            elif args[i] == 'comp_names':
                comp_names = kwargs[args[i]]
                
            elif args[i] == 'comp_norm':
                comp_norm = kwargs[args[i]]
                
            elif args[i] == 'lattice':
                sTopol['lattice'] = kwargs[args[i]]
                
            elif args[i] == 'shape':
                sTopol['shape'] = kwargs[args[i]]
                
            elif args[i] in ['topol', 'som_topol', 'sTopol']:
                sTopol = kwargs[args[i]]
                
            elif args[i] == 'neigh':
                neigh = kwargs[args[i]]
                
            elif args[i] in ['hexa','rect']:
                sTopol['lattice'] = kwargs[args[i]]
            elif args[i] in ['sheet','cyl','teroid']:
                sTopol['shape'] = kwargs[args[i]]
            elif args[i] in ['gaussian', 'cutgauss', 'ep', 'bubble']:
                neigh = kwargs[args[i]]
            else:
                argok = 0
        elif isinstance(args[i], dict) and 'type' in args[i]:
            if kwargs[args[i]]['type'] == 'som_topol':
                sTopol = kwargs[args[i]]
            else:
                argok = 0
        else:
            argok = 0

        if not argok:
            print(f'(som_map_struct) Ignoring invalid argument #{i + 1}')
        i =i+ 1

    # create the SOM
    # if sTopol['msize'] == 0:
    # # Initialize codebook as an empty array with dim columns
    #     codebook = np.empty((0, dim))
    # else:
    # Otherwise, generate codebook using random values
    
    np.random.seed(1337)
    codebook = np.random.rand(dim,np.prod(sTopol['msize']))
    codebook=codebook.T
    
    sTrain = som_set('som_train', *['time','mask'], **{'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mask':mask})  # Update time formatting
    # breakpoint()
    sMap = som_set('som_map', *['codebook', 'topol', 'neigh', 'labels', 'mask', 'comp_names', 'name', 'comp_norm', 'trainhist'], 
        **{'codebook': codebook,'topol': sTopol, 'neigh': neigh, 'labels': labels, 'mask': mask, 
            'comp_names': comp_names, 'name': name, 'comp_norm': comp_norm,'trainhist': sTrain})
    
    return sMap
