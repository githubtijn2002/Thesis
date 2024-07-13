# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:23:44 2024

@author: sdd380
"""

import numpy as np
import inspect
from som_set import som_set
from som_norm_variable import som_norm_variable
# from som_side_functions import *

def som_normalize(sD, *args):
    
    """
    SOM_NORMALIZE (Re)normalize data or add new normalizations.
       
    sS = som_normalize(sS,[method],[comps])               

       sS = som_normalize(sD) 
       sS = som_normalize(sS,sNorm) 
        D = som_normalize(D,'var')
       sS = som_normalize(sS,'histC',[1:3 10])

      Input and output arguments ([]'s are optional): 
       sS                The data to which the normalization is applied.
                        The modified and updated data is returned.
                (struct) data or map struct
                (matrix) data matrix (a matrix is also returned)
       [method]          The normalization method(s) to add/use. If missing, 
                        or an empty variable ('') is given, the 
                        normalizations in sS are used.
                (string) identifier for a normalization method to be added: 
                        'var', 'range', 'log', 'logistic', 'histD' or 'histC'. 
                (struct) Normalization struct, or an array of such. 
                        Alternatively, a map/data struct can be given 
                        in which case its '.comp_norm' field is used 
                        (see below).
                (cell array) Of normalization structs. Typically, the
                        '.comp_norm' field of a map/data struct. The 
                        length of the array must be equal to data dimension.
                (cellstr array) norm and denorm operations in a cellstr array
                        which are evaluated with EVAL command with variable
                        name 'x' reserved for the variable.
       [comps]  (vector) the components to which the normalization is
                        applied, default is [1:dim] ie. all components

    For more help, try 'type som_normalize' or check out online documentation.
    See also SOM_DENORMALIZE, SOM_NORM_VARIABLE, SOM_INFO.
    """
   
    csNorm = {}
    method= args[0]
    # breakpoint()
    if isinstance(sD, dict):  # Check if sD is a dictionary
        if 'data' in sD:  # Check if 'data' key is available
            D = sD['data']
        elif 'codebook' in sD:  # Check if 'codebook' key is available
            D = sD['codebook']
        else:
            raise ValueError("Invalid dictionary format. 'data' or 'codebook' key is missing.")
        dlen, dim = D.shape
    elif isinstance(sD, np.ndarray):  # Check if sD is a NumPy array
        D = sD
        dlen, dim = D.shape
    else:
        raise ValueError("Invalid input format. Expected a dictionary or a NumPy array.")
    # check arguments
    if  len(args)<2 or (isinstance(comps, str) and comps == 'all'):
        comps = list(range(1,dim+1))
    if not bool(comps):
        return
    
    # method
    csNorm = [[] for _ in range(dim)] 
    if len(args)<1 or not bool(method):
        method = ''
    else:
      # check out the given method (and if necessary, copy it for each specified component)  
      if isinstance(method,str):
          if method in {'var', 'range', 'log', 'histD', 'histC', 'logistic'}:
              sN = som_set('som_norm', *['method'],**{'method':method})
          else:
             raise ValueError( f"Unrecognized method: {method}")
          for i in comps:
              csNorm[i-1] = sN
    #check the size of csNorm is the same as data dimension
    if len(csNorm) !=dim:
        raise ValueError('Given number of normalizations does not match data dimension.')
          
   
    # Initialize
        
    struct_mode = isinstance(sD, dict)
    if struct_mode and 'type' in sD:
        
        if sD['type'] == 'som_map':
            D = sD.get('codebook', None)
        elif sD['type'] == 'som_data':
            D = sD.get('data', None)
        else:
            raise ValueError("Illegal struct.")
    else:
        D = sD['data']
   
       
    dlen, dim = D.shape if D is not None else (0, 0)
    
    # make sure all the current normalizations for current 
    # components have been done
    
    if struct_mode:
        alldone = True
        
        
        for i in comps:
            if sD['comp_norm'] and i < len(sD['comp_norm']):  # Check if sD['comp_norm'] is not empty and i is within its range
                for j in range(len(sD['comp_norm'][i])):
                    sN = sD['comp_norm'][i][j]
                    if sN[0]['status'] != 'done':
                        alldone = False
                        x, sN = som_norm_variable(D[:, i], sN, 'do')
                        D[:, i] = x
                        sD['comp_norm'][i][j] = sN
        
        if not method:
            if alldone:
                print("No 'undone' normalizations found. Data left unchanged.")
            else:
                print("Normalizations have been redone.")
    # action
    for i in comps:
        
        if isinstance(csNorm[i-1], dict) and bool(csNorm[i-1]):
            x, sN = som_norm_variable(D[:, i - 1], csNorm[i-1], 'do')
            D[:, i - 1] = x
            if not sD['comp_norm'][i - 1]:
                sD['comp_norm'][i - 1] = [sN]
            else:
                sD['comp_norm'][i - 1].append(sN)
             
        
                
            
        # output
        if struct_mode:
            
            if sD['type'] == 'som_map':
                sD['codebook'] = D
            elif sD['type'] == 'som_data':
                sD['data'] = D
            else:
                raise ValueError("Illegal struct.")
        else:
            sD = D

    return sD