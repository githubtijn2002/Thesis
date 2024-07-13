# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:05:03 2024

@author: sdd380
"""
import numpy as np
from sklearn.decomposition import PCA
from som_map_struct import som_map_struct
from som_lininit import som_lininit
from som_randinit import som_randinit
from som_topol_struct import som_topol_struct
from som_set import som_set
from som_train_struct import som_train_struct
from som_seqtrain import som_seqtrain
from som_sompaktrain import som_sompaktrain
from som_batchtrain import som_batchtrain
from som_quality import som_quality

def som_make(D, *args, **kwargs):
    # Parse input arguments
    # breakpoint()
    if isinstance(D, dict):
        data_name = D['name']
        comp_names = D['comp_names']
        comp_norm = D['comp_norm']
        D = D['data']
    else:
        data_name = 'D'
        pca = PCA(n_components=D.shape[1])
        pca.fit(D)
        comp_names = [f'PC{i+1}' for i in range(D.shape[1])]
        comp_norm = {i: pca.components_[i] for i in range(D.shape[1])}

    dlen, dim = D.shape
    
           
    # defaults
    mapsize = ''
    sM = som_map_struct(dim)
    
    sTopol = sM['topol']
    munits = np.prod(sTopol['msize'])  # should be zero
    mask = sM['mask']
    name = sM['name']
    neigh = sM['neigh']
    tracking = 1
    algorithm = 'batch'
    initalg = 'lininit'
    training = 'default'
    
    
    # args
    i = 0
    while i <= len(args)-1:
        
        argok = 1
        if isinstance(args[i], str):
            if args[i] == 'mask':
                mask = kwargs[args[i]]
                
            elif args[i] == 'munits':
                munits = kwargs[args[i]]
                
            elif args[i] == 'msize':
                sTopol['msize'] = kwargs[args[i]]
                munits = np.prod(sTopol['msize'])
                
            elif args[i] == 'mapsize':
                mapsize = kwargs[args[i]]
                
            elif args[i] == 'name':
                name = kwargs[args[i]]
                
            elif args[i] == 'comp_names':
                comp_names = kwargs[args[i]]
                
            elif args[i] == 'lattice':
                sTopol['lattice'] = kwargs[args[i]]
                
            elif args[i] == 'shape':
                sTopol['shape'] = kwargs[args[i]]
                
            elif args[i] in ['topol', 'som_topol', 'sTopol']:
                sTopol = kwargs[args[i]]
                munits = np.prod(sTopol['msize'])
                
            elif args[i] == 'neigh':
                neigh = kwargs[args[i]]
                
            elif args[i] == 'tracking':
                tracking = kwargs[args[i]]
                
            elif args[i] == 'algorithm':
                algorithm = kwargs[args[i]]
                
            elif args[i] == 'init':
                initalg = kwargs[args[i]]
                
            elif args[i] == 'training':
                training = kwargs[args[i]]
                
            elif args[i] in ['hexa', 'rect']:
                sTopol['lattice']= kwargs[args[i]]
            elif args[i] in ['sheet', 'cyl', 'toroid']:
                sTopol['shape'] = kwargs[args[i]]
            elif args[i] in ['gaussian','cutgauss','ep','bubble']:
                neigh = kwargs[args[i]]
            elif args[i] in ['seq','batch','sompak']:
                algorithm= kwargs[args[i]]
            elif args[i] in ['small','normal','big']:
                mapsize = kwargs[args[i]]
            elif args[i] in ['randinit','lininit']:
                initalg = kwargs[args[i]]
            elif args[i] in ['short','default','long']:
                training = kwargs[args[i]]
            else:
                argok = 0
        elif isinstance(args, dict) and 'type' in args:
            if args['type'] == 'som_topol':
                sTopol = args
            else:
                argok = 0
        else:
            argok = 0
    
        if not argok:
            print('(som_make) Ignoring invalid argument #', i + 1)
        i =i+ 1
    
    # Map size determination
    
    
    if bool(sTopol['msize']) or np.prod(sTopol['msize']) == 0:
        if tracking > 0:
            print('Determining map size...')
        if munits==0:
            sTemp = som_topol_struct(*['dlen'], **{'dlen': dlen})
            munits = np.prod(sTemp['msize'])
            if mapsize == 'small':
                munits = max(9, np.ceil(munits / 4))
            elif mapsize == 'big':
                munits *= 4
        sTemp= som_topol_struct(*['data', 'munits'],**{'data':D, 'munits': munits})    
        sTopol['msize'] = sTemp['msize']
        if tracking > 0:
            print(f"Map size [{sTopol['msize'][0]}, {sTopol['msize'][1]}]")
    
    
    sMap = som_map_struct(dim, *['sTopol','neigh','mask', 'name', 'comp_names', 'comp_norm'],
                                 **{'sTopol': sTopol, 'neigh': neigh, 'mask': mask, 'name': name, 'comp_names': comp_names, 'comp_norm': comp_norm})
    
    
    # function
    if algorithm == 'sompak':
        algorithm= 'seq'
        func = 'sompak'
    else:
        func=algorithm
       
    ## initialization
    if tracking > 0:
        print("Initialization...")
        
    if 'initalg' in locals():
        if initalg == 'randinit':
            sMap = som_randinit(D, *['sMap'],**{'sMap': sMap})
        elif initalg == 'lininit':
            sMap = som_lininit(D, *['sMap'],**{'sMap': sMap})
    
    sMap['trainhist'] = som_set(sMap['trainhist'], *['data_name'], **{'data_name': data_name})
    # make trainhist a list
    sMap['trainhist'] = [sMap['trainhist']]
    # training
    if tracking > 0:
        print(f"Training using {algorithm} algorithm...")
    
    # rough train
    if tracking > 0:
        print(f"Rough training phase...")
    
    sTrain= som_train_struct(*['sMap','dlen', 'algorithm', 'phase'],
                             **{'sMap': sMap,'dlen': dlen, 'algorithm': algorithm, 'phase': 'rough'})
    sTrain = som_set(sTrain, *['data_name'], **{'data_name': data_name})

    
    if isinstance(training, np.ndarray):
        sTrain['trainlen']=training
    else:
        if training == 'short':
            sTrain['trainlen'] = max((1, sTrain['trainlen']/4))
        elif training == 'long':
            sTrain['trainlen'] = sTrain['trainlen']*4
    
    if func == 'seq':
        sMap = som_seqtrain(sMap, D, *['sTrain','tracking', 'mask'], **{'sTrain': sTrain, 'tracking': tracking, 'mask': mask})
    elif func == 'sompak':
        sMap = som_sompaktrain(sMap, D, *['sTrain','tracking', 'mask'], **{'sTrain': sTrain, 'tracking': tracking, 'mask': mask})
    elif func == 'batch':
        sMap = som_batchtrain(sMap, D, *['sTrain','tracking', 'mask'], **{'sTrain': sTrain, 'tracking': tracking, 'mask': mask})
    
    
    # finetune
    if tracking > 0:
        print(f"finetuning phase...")
    sTrain = som_train_struct(*['sMap', 'dlen', 'phase'], **{'sMap': sMap, 'dlen': dlen, 'phase': 'finetune'})
    sTrain = som_set(sTrain, *['data_name', 'algorithm'], **{'data_name': data_name, 'algorithm': algorithm})
    
    
    if isinstance(training, np.ndarray):
        sTrain['trainlen']=training
    else:
        if training == 'short':
            sTrain['trainlen'] = max((1, sTrain['trainlen']/4))
        elif training == 'long':
            sTrain['trainlen'] = sTrain['trainlen']*4
    
    if func == 'seq':
        sMap = som_seqtrain(sMap, D, *['sTrain','tracking', 'mask'], **{'sTrain': sTrain, 'tracking': tracking, 'mask': mask})
    elif func == 'sompak':
        sMap = som_sompaktrain(sMap, D, *['sTrain','tracking', 'mask'], **{'sTrain': sTrain, 'tracking': tracking, 'mask': mask})
    elif func == 'batch':
        sMap = som_batchtrain(sMap, D, *['sTrain','tracking', 'mask'], **{'sTrain': sTrain, 'tracking': tracking, 'mask': mask})
    
    
    # quality
    if tracking >0:
        mqe , tge = som_quality(sMap, D)
        print(f'Final quantization error: {mqe:.3f}')
        print(f'Final topographic error:  {tge:.3f}')
    # Return trained SOM struct
    return sMap
