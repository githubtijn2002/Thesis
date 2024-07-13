# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:33:54 2024

@author: sdd380
"""
import numpy as np
from som_set import som_set 
import math

def som_train_struct(*args, **kwargs):
    
    #SOM_TRAIN_STRUCT Default values for SOM training parameters.
    #
    # sT = som_train_struct([[argID,] value, ...])
    #
    #  sTrain = som_train_struct('train',sM,sD);
    #  sTrain = som_train_struct('finetune','data',D); 
    #  sTrain = som_train_struct('previous',sT0);
    # 
    #  Input and output arguments ([]'s are optional): 
    #    [argID,  (string) Several default values depend on other SOM parameters
    #     value]  (varies) or on the proporties of a data set. See below for a
    #                      a list of required and optional arguments for
    #                      different parameters, and well as the list of valid 
    #                      argIDs and associated values. The values which are 
    #                      unambiguous can be given without the preceeding argID.
    #
    #    sT       (struct) The training struct.
    #
    # Training struct contains values for training and initialization
    # parameters. These parameters depend on the number of training samples,
    # phase of training, the training algorithm.
    # 
    # Here are the valid argument IDs and corresponding values. The values which
    # are unambiguous (marked with '*') can be given without the preceeding rgID.
    #  'dim'          (scalar) input space dimension
    #  'dlen'         (scalar) length of the training data
    #  'data'         (matrix / struct) the training data
    #  'munits'       (scalar) number of map units
    #  'msize'        (vector) map size
    #  'previous'     (struct) previous training struct can be given in 
    #                          conjunction with 'finetune' phase (see below) 
    #  'phase'       *(string) training phase: 'init', 'train', 'rough' or 'finetune'
    #  'algorithm'   *(string) algorithm to use: 'lininit', 'randinit', 'batch' or 'seq'
    #  'map'         *(struct) If a map struct is given, the last training struct
    #                          in '.trainhist' field is used as the previous training
    #                          struct. The map size and input space dimension are 
    #                          extracted from the map struct.
    #  'sTrain'      *(struct) a train struct, the empty fields of which are
    #                          filled with sensible values
    #
    # For more help, try 'type som_train_struct' or check out online documentation.
    # See also SOM_SET, SOM_TOPOL_STRUCT, SOM_MAKE.
    
    # DESCRIPTION
    #
    # This function is used to give sensible values for SOM training
    # parameters and returns a training struct. Often, the parameters
    # depend on the properties of the map and the training data. These are
    # given as optional arguments to the function. If a partially filled
    # train struct is given, its empty fields (field value is [] or '' or
    # NaN) are supplimented with default values.
    #
    # The training struct has a number of fields which depend on each other
    # and the optional arguments in complex ways. The most important argument 
    # is 'phase' which can be either 'init', 'train', 'rough' or 'finetune'.
    #
    #  'init'     Map initialization. 
    #  'train'    Map training in a onepass operation, as opposed to the
    #             rough-finetune combination.
    #  'rough'    Rough organization of the map: large neighborhood, big
    #             initial value for learning coefficient. Short training.
    #  'finetune' Finetuning the map after rough organization phase. Small
    #             neighborhood, learning coefficient is small already at 
    #             the beginning. Long training.
    #################################################
    
    ## check arguments

    # initial default structs
    sTrain = som_set('som_train')
    
    # initialize optional parameters
    dlen = np.nan
    msize = 0
    munits = np.nan
    sTprev =[]
    dim = np.nan
    phase = []
    
    i=0
    while i<= len(args)-1:
        argok = 1
        if isinstance(args[i], str):
            if args[i]=='dim':
                dim = kwargs[args[i]]
            elif args[i]=='dlen':
                dlen = kwargs[args[i]]
            elif args[i]== 'msize':
                msize = kwargs[args[i]]
            elif args[i] == 'munits':
                munits = kwargs[args[i]]
                msize = 0
            elif args[i]== 'phase':
                phase = kwargs[args[i]]
            elif args[i]== 'algorithm':
                sTrain['algorithm']= kwargs[args[i]]
            elif args[i]== 'mask':
                sTrain['mask']= kwargs[args[i]]
            elif args[i] == 'sMap' and 'type' in kwargs[args[i]]:
                if kwargs[args[i]]['type']=='som_train':
                    sT = kwargs[args[i]]
                    #...
                elif kwargs[args[i]]['type']== 'som_map':
                   if len(kwargs[args[i]]['trainhist'])>0:
                       if isinstance(kwargs[args[i]]['trainhist'],list):
                           sTprev = kwargs[args[i]]['trainhist'][-1]
                       else:
                           sTprev = kwargs[args[i]]['trainhist']
                       msize = kwargs[args[i]]['topol']['msize']
                   if bool(kwargs[args[i]]['neigh']) and not bool(sTrain['neigh']):
                       sTrain['neigh']= kwargs[args[i]]['neigh']
                   if bool(np.any(kwargs[args[i]]['mask'])) and not bool(np.any(sTrain['mask'])):
                       sTrain['mask']=kwargs[args[i]]['mask']
                elif kwargs[args[i]]['type']== 'som_train':
                    sTprev = kwargs[args[i]]
            # elif args[i] in ['previous', 'map']:
            #     if kwargs[args[i]]['type']== 'som_map':
                # continue in case it is needed!!!
            #         if len()
            elif args[i]== 'data':
                if isinstance(args[i], dict):
                    dlen, dim = kwargs[args[i]]['data'].shape
                else:
                    dlen, dim = kwargs[args[i]].shape
            elif args[i] == 'sTrain' and 'type' in kwargs[args[i]]:
                if kwargs[args[i]]['type']== 'som_train':
                    sT = kwargs[args[i]]
                    if bool(sT['algorithm']):
                        sTrain['algorithm'] = sT['algorithm']
                    if bool(sT['neigh']):
                        sTrain['neigh']= sT['neigh']
                    if bool(np.any(sT['mask'])):
                        sTrain['mask']= sT['mask']
                    if not np.isnan(sT['radius_ini']):
                        sTrain['radius_ini']=sT['radius_ini']
                    if not np.isnan(sT['radius_fin']):
                        sTrain['radius_fin']=sT['radius_fin']
                    if not np.isnan(sT['alpha_ini']):
                        sTrain['alpha_ini']=sT['alpha_ini']
                    if bool(sT['alpha_type']):
                        sTrain['alpha_type']= sT['alpha_type']
                    if not np.isnan(sT['trainlen']):
                        sTrain['trainlen']= sT['trainlen']
                    if bool(sT['data_name']):
                        sTrain['data_name']= sT['data_name']
                    if bool(sT['time']):
                        sTrain['time']= sT['time']
                elif kwargs[args[i]]['type']== 'som_map':
                    if len(kwargs[args[i]]['trainhist'])>1:
                        sTprev = [kwargs[args[i]]['trainhist'][-1]]
                        msize = kwargs[args[i]]['msize']
                    if bool(kwargs[args[i]]['neigh']) and not bool(sTrain['neigh']):
                        sTrain['neigh']= kwargs[args[i]]['neigh']
                    if bool(kwargs[args[i]]['mask']) and not bool(sTrain['mask']):
                        sTrain['mask']= kwargs[args[i]]['mask']
                elif kwargs[args[i]]['type']== 'som_topol':
                     msize = kwargs[args[i]]['msize']
                elif kwargs[args[i]]['type'] == 'som_data':
                    dlen, dim = kwargs[args[i]]['data'].shape
                else:
                    argok=0
        elif isinstance(args[i], dict) and 'type' in args[i]:
            later=[]
            # continue later
        else:
            argok = 0
        
        if argok == 0:
            print(f'(som_train_struct) Ignoring invalid argument #{i + 1}')
        i= i+1
    # dim
    
    if bool(sTprev) and np.isnan(dim):
        if isinstance(sTprev, list):
            dim = len(sTprev[-1]['mask'])
        else:
            dim = len(sTprev['mask'])
    
    # mask
    if not bool(np.any(sTrain['mask'])) and not np.isnan(dim):
        sTrain['mask'] = np.ones((dim, 1))
    
    # msize, munits
    if msize==0 or not bool(msize):
        if np.isnan(munits):
            msize= (10,10)
        else:
            s= round(math.sqrt(munits))
            msize = (s, round(munits/s))
    munits = np.prod(msize)
    
    
    ## action
    # previous training     
    prevalg = []
    if bool(sTprev):
        if isinstance(sTprev, dict):
            if 'init' in sTprev['algorithm']:
                prevalg = 'init'
            else:
                prevalg = sTprev['algorithm']
        elif isinstance(sTprev, list):
            if 'init' in sTprev[-1]['algorithm']:
                prevalg = 'init'
            else:
                prevalg = sTprev[-1]['algorithm']
    
    # first determine phase
    if not bool(phase):
        if sTrain['algorithm'] in ['lininit', 'randinit']:
            phase = 'init'
        elif sTrain['algorithm'] in ['batch', 'seq', '']:
            if not bool(sTprev):
                phase = 'rouhg'
            elif prevalg == 'init':
                phase = 'rough'
            else:
                phase = 'finetune'
        else:
            phase = 'train'
    
    # then determine algorithm
    if not bool(sTrain['algorithm']):
        if phase == 'init':
            sTrain['algorithm']= 'lininit'
        elif prevalg in ['init', '']:
            sTrain['algorithm'] = 'batch'
        else:
            sTrain['algorithm']= sTprev['algorithm']
    
    # mask
    
    if not bool(np.any(sTrain['mask'])):
        if bool(sTprev):
            sTrain['mask']= sTprev['mask']
        elif not np.isnan(dim):
            sTrain['mask']= np.ones((dim, 1))
    
    # neigborhood function
    if not bool(sTrain['neigh']):
        if bool(sTprev) and bool(sTprev['neigh']):
            sTrain['neigh']= sTprev['neigh']
        else:
            sTrain['neigh']=='gaussian'
        
    if phase == 'init':
        sTrain['alpha_ini']= np.nan         
        sTrain['alpha_type']= ['']
        sTrain['radius_ini']= np.nan
        sTrain['radius_fin']= np.nan
        sTrain['trainlen']= np.nan
        sTrain['neigh']= ['']
    else:
        mode = '-'.join([phase, sTrain['algorithm']])
    
        # learning rate
        if np.isnan(sTrain['alpha_ini']):
            if sTrain['algorithm']=='batch':
                sTrain['alpha_ini']=np.nan
            else:
                if phase in ['train', 'rough']:
                    sTrain['alpha_ini'] = 0.5
                elif phase == 'finetune':
                    sTrain['alpha_ini'] = 0.05
                
        if not bool(sTrain['alpha_type']):
            if bool(sTprev) and bool(sTprev['alpha_type']) and sTrain['algorithm'] !='batch':
                sTrain['alpha_type'] = sTprev['alpha_type']
            elif sTrain['algorithm'] == 'seq':
                sTrain['alpha_type'] = 'inv'
        
        # radius
        
        ms = max(msize)
        if np.isnan(sTrain['radius_ini']):
            if isinstance(sTprev, dict):
                if not bool(sTprev) or sTprev['algorithm']== 'randinit':
                    sTrain['radius_ini']= max(1, math.ceil(ms/4))
                elif sTprev['algorithm']=='lininit' or np.isnan(sTprev['radius_fin']):
                    sTrain['radius_ini']= max(1, math.ceil(ms/8))
                else:
                    sTrain['radius_ini']= sTprev['radius_fin']
            elif isinstance(sTprev, list):
                if not bool(sTprev) or sTprev[-1]['algorithm']== 'randinit':
                    sTrain['radius_ini']= max(1, math.ceil(ms/4))
                elif sTprev[-1]['algorithm']=='lininit' or np.isnan(sTprev[-1]['radius_fin']):
                    sTrain['radius_ini']= max(1, math.ceil(ms/8))
                else:
                    sTrain['radius_ini']= sTprev[-1]['radius_fin']
        
        if np.isnan(sTrain['radius_fin']):
            if phase == 'rough':
                sTrain['radius_fin']= max(1, sTrain['radius_ini']/4)
            else:
                sTrain['radius_fin']=1
        
        
        # trainlen
        if np.isnan(sTrain['trainlen']):
            mpd = munits/dlen
            if np.isnan(mpd):
                mpd = 0.5
            if phase == 'train':
                sTrain['trainlen']= math.ceil(50 * mpd)
            elif phase == 'rough':
                sTrain['trainlen']= math.ceil(10 * mpd)
            elif phase == 'finetune':
                sTrain['trainlen']= math.ceil(40 * mpd)
            sTrain['trainlen'] = max(1, sTrain['trainlen'])
        
    return sTrain