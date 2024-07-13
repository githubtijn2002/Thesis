# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:00:56 2024

@author: sdd380
"""
import numpy as np
from som_norm_variable import som_norm_variable

def som_denormalize(sD, *args):
    
    # DESCRIPTION
    #
    # This function is used to undo normalizations of data structs/sets. If a
    # data/map struct is given, all normalizations in the '.comp_norm' field are
    # undone and, thus, the values in the original data context are returned. If
    # a matrix is given, the normalizations to undo must be given as the second
    # argument. SOM_DENORMALIZE actually uses function SOM_NORM_VARIABLE to
    # handle the normalization operations, and only handles the data struct/set
    # specific stuff itself.
    #
    # Normalizations are always one-variable operations. In the data and map
    # structs the normalization information for each component is saved in the
    # '.comp_norm' field, which is a cell array of length dim. Each cell
    # contains normalizations for one vector component in a
    # struct array of normalization structs. Each component may have different
    # amounts of different kinds of normalizations. Typically, all
    # normalizations are either 'undone' or 'done', but in special situations
    # this may not be the case. The easiest way to check out the status of the
    # normalizations is to use function SOM_INFO, e.g. som_info(sS,3)
    
    # sD
    struct_mode = isinstance(sD, dict)
    
    if struct_mode==1:
        if sD['type']== 'som_map':
            D = sD['codebook']
        elif sD['type']== 'som_data':
            D = sD['data']
        else:
            raise ValueError('illegal struct')
    else:
        D = sD
    dlen, dim = D.shape
    
    comps = np.arange(dim)
    remove_tag = 0
    if struct_mode == 1:
        sNorm = sD['comp_norm']
    else:
        sNorm = ([])
    
    i=0
    while i <=len(args)-1:
        argok =1
        if isinstance(args[i],str):
            if args[i]== 'comps':
                comps = args[i]
            elif args[i] in {'norm', 'sNorm', 'som_norm'}:
                sNorm = args[i]
            elif args[i] == 'remove':
                remove_tag = 1
            else:
                argok = 0
        elif isinstance(args[i], np.ndarray):
            comps=args[i]
        elif isinstance(args[i], dict):
            sNorm = args[i]
        elif isinstance(args[i], list):
            sNorm = args[i]
        else:
            argok = 0
        if argok == 0:
            print(print(f'(som_denormalize) Ignoring invalid argument #{i}'))
        i = i+1
       
    # check comps
    if isinstance(comps, str):
        comps = np.arange(dim)
    if comps.size==0:
        return
   
    # sNorm
    # check out the given normalisation and if necessary copy it for each specified component
    if isinstance(sNorm, dict):
        if sNorm['type'] in {'som_map', 'som_data'}:
           csNorm = sNorm['comp_norm']
        elif sNorm['type'] == 'som_norm':
            for i in comps:
                csNorm[i] = sNorm
        else:
            raise ValueError('Illegal struct for sNorm')
    elif isinstance(sNorm, list):
         csNorm = sNorm
    else:
         raise ValueError('Illegar value for sNorm')
        
    # check if csNorm and comps possibly agree
    if max(comps) > len(csNorm):
         raise ValueError('Given normalizations does not match the components.')
    if len(csNorm) != dim:
         raise ValueError('Given normalizations does not match data dimension.') 

    
    # action
    for i in comps:
        leng = len(csNorm[i])
        if isinstance(csNorm,tuple):
            for j in range(leng, -1, -1):
                sN = csNorm[i][j]
                if struct_mode ==1:
                    if sN['status']=='done':
                        x, sN = som_norm_variable(D[:,i], sN, 'undo')
                        D[:,i] = x
                        csNorm[i][j] = sN
                else:
                    D[:,i] = som_norm_variable(D[:,i], sN, 'undo' )
        else:
            sN = csNorm[i][0]
            
            if struct_mode ==1:
                if sN['status']=='done':
                    x, sN = som_norm_variable(D[:,i], sN, 'undo')
                    D[:,i] = x
                    csNorm[i][j] = sN
            else:
                D[:,i], _ = som_norm_variable(D[:,i], sN, 'undo' )
                        
                       
    # remove normalizations
    if struct_mode==1 and remove_tag==1:
        for i in comps:
            csNorm[i][0]=([])
    
    ### output
    
    if struct_mode==1:
        if sD['type']=='som_map':
            sD['codebook'] = D
        elif sD['type'] == 'som_data':
            sD['data'] = D
        else:
            raise ValueError('Illegal struct')
        sD['comp_norm']= csNorm
    else:
        sD = D
    
    return sD
                
            
           
        
    
         