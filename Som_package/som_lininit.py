# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:25:45 2024

@author: sdd380
"""

import numpy as np
from som_topol_struct import som_topol_struct 
from som_map_struct import som_map_struct
from som_set import som_set
from som_train_struct import som_train_struct
from som_unit_coords import som_unit_coords
from datetime import datetime

def som_lininit(D, *args, **kwargs):
    
    if isinstance(D, dict):
        data_name = D['name']
        comp_names = D['comp_names']
        comp_norm = D['comp_norm']
        D= D['data']
        struct_mode =1
    else:
        data_name = 'D'
        structmode =0
        
    dlen, dim = D.shape
    
    sMap = []
    sTopol = som_topol_struct()
    sTopol['msize']=0
    munits= None
    
    i=0
    while i <= len(args)-1:
       argok=1
       if isinstance(args[i], str):
           
           if args[i]== 'munits':
              munits= kwargs[args[i]]
              sTopol['msize']=0
              
           elif args[i] == 'msize':
               sTopol['msize']= kwargs[args[i]]
               munits = np.prod(sTopol['msize'])
               
           elif args[i]== 'lattice':
               sTopol['lattice']= kwargs[args[i]]
               
           elif args[i]=='shape':
               sTopol['shape'] = kwargs[args[i]]
               
           elif args[i] in ['som_topol','sTopol', 'topol']:
               sTopol= kwargs[args[i]]
               
           elif args[i] in ['som_map','sMap','map']:
               sMap = kwargs[args[i]]
               sTopol = sMap['topol']
               
           elif args[i] in ['hexa','rect']:
                  sTopol['lattice']= kwargs[args[i]]
           elif args[i] in ['sheet','cyl','toroid']:
               sTopol['shape']= kwargs[args[i]]
           else:
               argok=0
       elif isinstance(args[i], dict) and 'type'in args[i]: 
           if args[i]['type']=='som_topol':
               sTopol = kwargs[args[i]]
           elif args[i]['type']== 'som_map':
               sMap = kwargs[args[i]]
               sTopol= sMap['topol']
           else:
               argok=0
       else:
           argok=0
            
       if argok ==0:
           print('(som_topol_struct) Ignoring invalid argument #' + str(i))
       i = i+1
    
       
    if len(sTopol['msize'])==1:
        sTopol["msize"].append(1)
       
    if bool(sMap):
        munits, dim2 = sMap['codebook'].shape
    
    if dim2 != dim:
        raise ValueError('Map and data must have the same dimension.')

           
   # create map
   # map struct
    
    if bool(sMap):
        sMap = som_set(sMap, *['topol'], **{'topol': sTopol})
    
    else:
        if not np.prod(sTopol['msize']):
            if np.isnan(munits):
                sTopol = som_topol_struct(*['data', 'sTopol'], **{'data': D, 'sTopol': sTopol})
            else:
                sTopol = som_topol_struct(*['data', 'munits', 'sTopol'], **{'data': D,'munits': munits, 'sTopol': sTopol})
            
            sMap = som_map_struct(dim, args, kwargs)
    
    if structmode ==1:
        sMap = som_set(sMap, *['comp_names', 'comp_norm'], **{'comp_names': comp_names, 'comp_norm': comp_norm})
           
    
    
    ## initialization
    # train struct
    sTrain = som_train_struct(*['algorithm'],**{'algorithm':'lininit'})
    sTrain = som_set(sTrain, *['data_name'], **{'data_name': data_name})
    
    msize = np.array(sMap['topol']['msize'])
    mdim = len(msize)
    munits = np.prod(msize)
    
    dlen, dim = D.shape
    if dlen<2:
        raise ValueError('Linear map initialization requires at least two NaN-free samples.')
        return
    
    
    
    ## compute principle components
    if dim > 1 and sum(1 for x in msize if x > 1) > 1:
        # Calculate mdim largest eigenvalues and their corresponding eigenvectors
        # Autocorrelation matrix
        A = np.zeros((dim, dim))
        me = np.zeros(dim)
        
        
        # me = np.mean(D, axis=0)  # Mean of finite values
        # D = D - me
        for i in range(dim):
            # Compute mean of finite (non-NaN) values in the i-th column
            me[i] = np.nanmean(D[:, i])  # Use np.nanmean to ignore NaNs in the computation
           

            # Subtract the mean from each element in the i-th column
            D[:, i] = D[:, i] - me[i]
          
        for i in range(dim):
            for j in range(i, dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = np.sum(c) / len(c)
                A[j, i] = A[i, j]
        
        
        # Take mdim first eigenvectors with the greatest eigenvalues
        # Step 1: Compute eigenvectors and eigenvalues
        eigval, V = np.linalg.eig(A)
        
        # Step 2: Extract eigenvalues and sort
        ind = np.argsort(eigval)[::-1]  # Sort eigenvalues in descending order
        eigval = eigval[ind]
        V = V[:, ind]
        
        # Step 3: Select top mdim eigenvalues and eigenvectors
        mdim = 2  # Example: Selecting top 2 eigenvalues and eigenvectors
        eigval = eigval[:mdim]
        V = V[:, :mdim]
        
        # Step 4: Normalize eigenvectors and scale by square root of eigenvalues
        for i in range(mdim):
            norm = np.linalg.norm(V[:, i])
            V[:, i] = (V[:, i] / norm) * np.sqrt(eigval[i])             
        
               
        # S, V = np.linalg.eig(A)
        # ind = np.argsort(S)[::-1]  # Sort indices in descending order
        # S = S[ind] # S=eigval
        # V = V[:, ind]
        # S = np.diag(S)
        # for i in range(V.shape[1]):
        #     if V[0, i] < 0:
        #         V[:, i] = -V[:, i]
        # eigval = np.diag(S)[:mdim]
        # V = V[:, :mdim]

        
        # # Normalize eigenvectors to unit length and multiply them by corresponding (square-root-of-)eigenvalues
        # for i in range(mdim):
        #     V[:, i] = (V[:, i] / np.linalg.norm(V[:, i])) * np.sqrt(eigval[i])
    
    else:
        me = np.zeros(dim)
        V = np.zeros(dim)
        
        for i in range(dim):
            inds = np.where(~np.isnan(D[:, i]))[0]
            me[i] = np.nanmean(D[inds, i])
            V[i] = np.nanstd(D[inds, i])
    
    ## initialize codebook vectors
    
    if dim >1:
        sMap['codebook']= np.tile(me, (munits, 1))
        Coords = som_unit_coords(msize, *['lattice', 'shape'], **{'lattice': 'rect', 'shape': 'sheet'})
        cox = Coords[:,0].copy()
        Coords[:,0] = Coords[:,1]
        Coords[:,1] = cox.copy()
        for i in range(mdim):
            ma = max(Coords[:,i])
            mi = min(Coords[:,i])
            if ma > mi:
                Coords[:,i] = (Coords[:,i] - mi)/(ma-mi)
            else:
                Coords[:,i] = 0.5
        Coords = (Coords - 0.5)*2
        for n in range(munits):
            for d in range(mdim):
                sMap['codebook'][n,:] = sMap['codebook'][n,:] + Coords[n,d] * V[:,d]
    else:
        sMap['codebook'] = (np.arange(munits) / (munits - 1)) * (max(D) - min(D)) + min(D)

            
    
    # training struct
    sTrain = som_set(sTrain, *['time'], **{'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    sMap['trainhist'] = sTrain.copy()
        
        
        
        
    return sMap