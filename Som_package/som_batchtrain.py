# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:19:39 2024

@author: sdd380
"""

import numpy as np
import inspect
from som_set import som_set
from som_train_struct import som_train_struct
from som_unit_dists import som_unit_dists
import time
import math
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from som_set import som_set
from datetime import datetime

def som_batchtrain(sMap, D, *args, **kwargs):
    
    ############# DETAILED DESCRIPTION ########################################
    #
    # som_batchtrain
    #
    # PURPOSE
    #
    # Trains a Self-Organizing Map using the batch algorithm.
    # Trains the given SOM (sM or M above) with the given training data
    # (sD or D) using batch training algorithm.  If no optional arguments
    # (argID, value) are given, a default training is done. Using optional
    # arguments the training parameters can be specified. Returns the
    # trained and updated SOM and a train struct which contains
    # information on the training.
    #
    # REFERENCES
    #
    # Kohonen, T., "Self-Organizing Map", 2nd ed., Springer-Verlag, 
    #    Berlin, 1995, pp. 127-128.
    # Kohonen, T., "Things you haven't heard about the Self-Organizing
    #    Map", In proceedings of International Conference
    #    on Neural Networks (ICNN), San Francisco, 1993, pp. 1147-1156.
    
    # REQUIRED INPUT ARGUMENTS
    #
    #  sM          The map to be trained. 
    #     (struct) map struct
    #     (matrix) codebook matrix (field .data of map struct)
    #              Size is either [munits dim], in which case the map grid 
    #              dimensions (msize) should be specified with optional arguments,
    #              or [msize(1) ... msize(k) dim] in which case the map 
    #              grid dimensions are taken from the size of the matrix. 
    #              Lattice, by default, is 'rect' and shape 'sheet'.
    #  D           Training data.
    #     (struct) data struct
    #     (matrix) data matrix, size [dlen dim]
    
    # check arguments
    num_args = len(args)
    if num_args < 2:
        raise ValueError(f"Too few arguments: expected at least 2, got {num_args}")
    if num_args > float('inf'):
        raise ValueError(f"Too many arguments: expected at most inf, got {num_args}")
        
    # map
    struct_mode = isinstance(sMap, dict)
    if struct_mode == True:
        sTopol = sMap['topol']
    else:
        dlens, ndims = sMap.shape
        if ndims(sMap) > 2:
            si= sMap.shape
            dim = si[0, -1]
            msize = si[0:-1]
    munits, dim = sMap['codebook'].shape      
    
    # data
    if isinstance(D, dict):
        data_name = D['name']
        D = D['data']
    else:
        sig = inspect.signature(som_batchtrain)
        params = sig.parameters
        param_list = list(params.keys())
        data_name = param_list[1]
   
    nonempty = np.where(np.sum(np.isnan(D), axis=1) < dim)[0]
    D = D[nonempty,:]
    dlen, ddim = D.shape
    if dim != ddim:
        raise ValueError('Map and data input space dimensions disagree.')
    
    # args
    sTrain = som_set('som_train', *['algorithm', 'neigh', 'mask', 'data_name'], 
                     **{'algorithm': 'batch', 'neigh': sMap['neigh'], 'mask': sMap['mask'], 'data_name': data_name})
    
    radius = []
    tracking = 1
    weights = np.ones(1)
    
    i=0
    while i<= len(args)-1:
        argok = 1
        if isinstance(args[i], str):
            if args[i]== 'msize':
                sTopol['msize'] = kwargs[args[i]]
            elif args[i] == 'lattice':
                sTopol['lattice']= kwargs[args[i]]
            elif args[i] == 'shape':
                sTopol['shape']= kwargs[args[i]]
            elif args[i] == 'mask':
                sTrain['mask']= kwargs[args[i]]
            elif args[i] == 'neigh':
                sTrain['neigh']= kwargs[args[i]]
            elif args[i] == 'trainlen':
                sTrain['trainlen'] = kwargs[args[i]]
            elif args[i] == 'tracking':
                tracking = kwargs[args[i]]
            elif args[i] == 'weights':
                weights = kwargs[args[i]]
            elif args[i] == 'radius_ini':
                sTrain['radius_ini'] = kwargs[args[i]]
            elif args[i] == 'radius_fin':
                sTrain['radius_fin'] = kwargs[args[i]]
            elif args[i] == 'radius':
                l = len(kwargs[args[i]])
                if l==1:
                    sTrain['radius_ini']=kwargs[args[i]]
                else:
                    sTrain['radius_ini']= kwargs[args[i][1]]
                    sTrain['radius_fin']= kwargs[args[i][-1]]
                    if l>2:
                        radius = kwargs[args[i]]
            elif args[i] in {'sTrain','train','som_train'}:
                sTrain = kwargs[args[i]]
            elif args[i] in {'topol','sTopol','som_topol'}:
                sTopol = kwargs[args[i]]
                if np.prod(sTopol['msize']) != munits:
                    raise ValueError('Given map grid size does not match the codebook size.')
            # unambiguous values
            elif args[i] in {'hexa','rect'}:
                sTopol['lattice'] = kwargs[args[i]]
            elif args[i] in {'sheet','cyl','toroid'}:
                sTopol['shape'] = kwargs[args[i]]
            elif args[i] in {'gaussian','cutgauss','ep','bubble'}:
                sTrain['neigh'] = kwargs[args[i]]
            else:
                argok=0
        elif isinstance(args[i], dict) and 'type' in kwargs[args[i]]:
            if kwargs[args[i]]['type'] == 'som_topol':
                sTopol= kwargs[args[i]]
                if np.prod(sTopol['msize']) != munits:
                    raise ValueError('Given map grid size does not match the codebook size.')
            elif kwargs[args[i]]['type'] == 'som_train':
                sTrain = kwargs[args[i]]
            else:
                argok =0
        else:
            argok=0
        
        if argok == 0:
            print("(som_batchtrain) Ignoring invalid argument #" + str(i))
        i=i+1
        
    # take only weights of non-empty vectors
    if len(weights) > dlen:
        weights = weights[nonempty]
    
    # trainlen
    if bool(radius):
        sTrain['trainlen'] = len(radius)
    
    # check topology
    if bool(struct_mode):
        if sTopol['lattice'] != sMap['topol']['lattice'] or sTopol['shape'] != sMap['topol']['shape'] or np.any(sTopol['msize'] != sMap['topol']['msize']):
            print("Changing the original map topology")
    
    sMap['topol'] = sTopol
    
    # complement the training struct
    sTrain = som_train_struct(*['sTrain','sMap', 'dlen'],**{'sTrain': sTrain,'sMap': sMap, 'dlen': dlen})
    if not bool(np.any(sTrain['mask'])):
        sTrain['mask'] = np.ones((dim,1))
        
    #### INITIALIZE
    M = sMap['codebook']
    mask = sTrain['mask']
    trainlen = sTrain['trainlen']
    
    # neigbourhood radius
    if trainlen == 1:
        radius = np.array([sTrain['radius_ini']])
    elif len(radius)<=2:
        r0 = sTrain['radius_ini']
        r1 = sTrain['radius_fin']
        radius = r1 + (np.flip(np.arange(trainlen))/ (trainlen - 1)) * (r0 - r1)
    
    # distance between map units in the output space
    #  Since in the case of gaussian and ep neighborhood functions, the 
    #  equations utilize squares of the unit distances and in bubble case
    #  it doesn't matter which is used, the unitdistances and neighborhood
    #  radiuses are squared.
    
    Ud = som_unit_dists(sTopol)
    
    Ud = Ud**2
    radius = radius **2
    
    #ero neighborhood radius may cause div-by-zero error
    eps = np.finfo(float).eps
    radius[radius == 0] = eps
    
   
    Known =  (~np.isnan(D)).astype(int)
    D[~Known.astype(bool)] = 0
    W1 = (mask * np.ones((1, dlen)) * Known.T)
    # constant matrices
    WD = 2 * np.dot(np.diag(mask.flatten()), D.T)
    dconst = np.dot((D**2),mask).T
               
    start = time.time()
    qe = np.zeros((trainlen,1))  

    ##### Action
    # With the 'blen' parameter you can control the memory consumption 
    # of the algorithm, which is in practive directly proportional
    # to munits*blen. If you're having problems with memory, try to 
    # set the value of blen lower. 
    blen = min((munits,dlen))
    
    # reserve some space
    # ddists = np.zeros(dlen) # changed here
    # bmus = np.zeros(dlen, dtype=int)
    
    for t in range(0, trainlen):
      
      ddists = np.zeros(dlen) # changed here
      bmus = np.zeros(dlen, dtype=int)
      i0 = 0
      while i0 + 1 <= dlen:
        inds = np.arange(i0, min(dlen, i0 + blen))
        i0 += blen
        
        # Calculate Dist
        
        Dist = np.square(M) @ W1[:, inds] - M @ WD[:, inds]
        ddists[inds] = np.min(Dist, axis=0)
        bmus[inds] = np.argmin(Dist, axis=0) 

      # tracking
      if tracking >0:
          ddists = ddists + dconst
          ddists[ddists<0] = 0 
          qe[t] = np.mean(np.sqrt(ddists))
          trackplot(M,D,tracking,start,t,qe)

      H = np.zeros_like(Ud)
    
      if sTrain['neigh']=='bubble':
           H[Ud <= radius[t]] = 1
      elif sTrain['neigh']=='gaussian':
           H = np.exp(-Ud / (2 * radius[t]))
      elif sTrain['neigh']=='cutgauss':
           H = np.exp(-Ud / (2 * radius[t])) * (Ud <= radius[t])
      elif sTrain['neigh']=='ep':
           H = (1 - Ud / radius[t]) * (Ud <= radius[t])
      else:
          raise ValueError(f"Unknown neighborhood type: {sTrain['neigh']}")
      # update 

      
      
      weights1=np.full(len(bmus), weights)  
      P = coo_matrix((weights1, (bmus, np.arange(len(bmus)))), shape=(munits, dlen))
      S = np.dot(H , P @ D)
      A = np.dot(H, P @ Known)
      
      # only update units for which the "activation" is nonzero
      nonzero = np.where(A>0)
      M = np.zeros_like(A)
      M[nonzero[0]] = S[nonzero[0]]/A[nonzero[0]]
    
    
    # Build / clean up the return arguments
    if tracking > 0:
        print('\n')
    # update structures
    sTrain = som_set(sTrain, *['time'], **{'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    if struct_mode == 1:
        sMap = som_set(sMap, *['codebook', 'mask', 'neigh'], **{'codebook': M, 'mask': sTrain['mask'], 'neigh': sTrain['neigh']})
        sMap['trainhist'].append(sTrain) 
    else:
        sMap = np.reshape(M, orig_size)
    
   
    return sMap    
    

def trackplot(M,D,tracking,start,n,qe):
    

    
    l=len(qe)
    elap_t = time.time() - start

    # Total time
    tot_t = elap_t * l / (n+1)
    
    # Print training progress
    print(f'\rTraining: {elap_t:.0f} / {tot_t:.0f} s', end='')
    
    # Switch statement equivalent
    tracking = 2  # Example value for tracking
    
    if tracking == 1:
        pass  # Case 1: No action
    elif tracking == 2:
        # Plot quantization error after each epoch
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, n + 1), qe[:n], 'b-', label='First part')
        plt.plot(range(n + 1, l + 1), qe[n:l], 'r-', label='Second part')
        plt.title('Quantization error after each epoch')
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)  # Needed for interactive plotting
        plt.show()
    else:
        # Default case: subplot visualization
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(range(1, n + 1), qe[:n], 'b-', label='First part')
        plt.plot(range(n + 1, l + 1), qe[n:l], 'r-', label='Second part')
        plt.title('Quantization error after each epoch')
        plt.legend()
        plt.grid(True)
    
        plt.subplot(2, 1, 2)
        # Example data M and D (replace with your actual data)
        M = np.random.rand(10, 2)
        D = np.random.rand(10, 2)
        plt.plot(M[:, 0], M[:, 1], 'ro', label='Map units (o)')
        plt.plot(D[:, 0], D[:, 1], 'b+', label='Data vectors (+)')
        plt.title('First two components of map units (o) and data vectors (+)')
        plt.legend()
        plt.grid(True)
    
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Needed for interactive plotting
    
        plt.show() 