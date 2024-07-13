# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:05:18 2024

@author: sdd380
"""

# som set

import numpy as np
import math


def som_set(sS, *args, **kwargs):
    ok = []
    msgs = []
    
    
    if isinstance(sS, str):
        if sS == 'som_map':
            sS = {
                'type': 'som_map',
                'codebook': np.array([]),
                'topol': som_set('som_topol'),
                'labels': [],
                'neigh': 'gaussian',
                'mask': np.array([]),
                'trainhist': [],
                'name': '',
                'comp_names': [''],
                'comp_norm': []
                }
        elif sS == 'som_data':
            sS = {
                'type': 'som_data',
                'data': np.array([]),
                'labels': [],
                'name': '',
                'comp_names': [''],
                'comp_norm': [],
                'label_names': []
            }
        elif sS == 'som_topol':
            sS = {
                'type': 'som_topol',
                'msize': 0,
                'lattice': 'hexa',
                'shape': 'sheet'
            }
        elif sS == 'som_train':
            sS = {
                'type': 'som_train',
                'algorithm': '',
                'data_name': '',
                'neigh': 'gaussian',
                'mask': np.array([]),
                'radius_ini': np.nan,
                'radius_fin': np.nan,
                'alpha_ini': np.nan,
                'alpha_type': 'inv',
                'trainlen': np.nan,
                'time': ''
            }
        elif sS == 'som_norm':
            sS = {
                'type': 'som_norm',
                'method': 'var',
                'params': [],
                'status': 'uninit'
            }
        
        elif sS == 'som_grid':
            sS = {
                'type': 'som_grid',
                'lattice': 'hexa',
                'shape': 'sheet',
                'msize': [1, 1],
                'coord': [],
                'line': '-',
                'linecolor': [.9, .9, .9],
                'linewidth': 0.5,
                'marker': 'o',
                'markersize': 6,
                'markercolor': 'k',
                'surf': [],
                'label': [],
                'labelcolor': 'g',
                'labelsize': 12
            }
        else:
            ok.append(0)
            msgs.append(['Unrecognized struct type: {}'.format(sS)])
            sS = None
            # return sS, ok, msgs
    elif isinstance(sS, dict) and len(args)==0:
        # Check all fields
        fields = sS.keys()
        if 'type' not in fields:
            raise ValueError("The struct has no 'type' field.")
        k = 0
        for field in fields:
            content = sS[field]
            if field != 'type':
                args.append(field)
                kwargs.append(content)
    # elif isinstance(sS, list):
    #     fields = sS[0].keys()
    #     if 'type' not in fields:
    #         raise ValueError("The struct has no 'type' field.")

    #     # Iterate through fields and update args and kwargs
    #     for field in fields:
    #         if field != 'type':
    #             # Append the field name to args if not already present
    #             if field not in args:
    #                 args += (field,)  # Tuple concatenation to add a single element to tuple
    #             # Add the field content to kwargs
    #             kwargs[field] = sS[0][field]
    #     sS = sS[0]
    ## set field values
    p = len(args)
    ok = np.ones((p, 1))
    msgs = [None] * p
    for i in range(0, p):
        field = args[i]
        content = kwargs[field]
        msgs = ['']
        isok = 0
        
        # si = content.shape
        if isinstance(content, list) and len(content) == 0:
            si = (0, 0)
        elif isinstance(content, np.ndarray) and content.shape == 0:
            si = (0, 0)
        elif isinstance(content, np.ndarray) and content.shape != 0:
            si = content.shape
        elif isinstance(content,str):
            si = (1,len(content))
       
        elif isinstance(content,dict) and len(content) >0:
            si = (0,len(content))
        else:
            si = np.shape(content)
            
            
        isscalar = np.prod(si)==1
        isvector = np.sum(np.array(si)>1)==1
        isrowvector = isvector & si==1
        if isinstance(content, np.ndarray):
            iscomplete = np.all(~np.isnan(content))
            ispositive = np.all(content > 0)
            isinteger = np.all(np.asarray(content) == np.ceil(np.asarray(content)))
            isrgb = (np.all(content > 0) & np.all(content <= 1)) & content.shape[1]==3
        
        if sS['type']== 'som_train':
            if field== 'algorithm':
                if not isinstance(content,str):
                    print("'algorithm'' should be a string.'")
                else:
                    sS['algorithm'] = content
                    isok=1
            elif field== 'data_name':
                if not isinstance(content,str):
                    print("'data name'' should be a string.'")
                else:
                    sS['data_name'] = content
                    isok=1
            elif field=='neigh':
                if not isinstance(content,str):
                    print("'neigh'' should be a string.'")
                elif bool(content) and content != 'gaussian' and content !='ep' and content != 'cutgauss' and content != 'bubble':
                    print("Unknown neighborhood function: content")
                    sS['neigh']= content
                    isok=1
                else:
                    sS['neigh'] = content
                    isok=1
            elif field == 'mask':
                if len(content)==1:
                    [list(row) for row in zip(*content)]
                dim = len(content)
                if not isinstance(content, np.ndarray) or content.shape != (dim,1):
                    print("'mask'' should be a column vector (size dim x 1).")
                else:
                    sS['mask']= content
                    isok=1
            elif field == 'radius_ini':
                if not isinstance(content, np.ndarray) or 'isscalar' not in locals():
                    print("radius_ini'' should be a scalar")
                else:
                    sS['radius_ini']= content
                    isok=1
            elif field== 'radius_fin':
                if not isinstance(content, np.ndarray) or 'isscalar' not in locals():
                    print("'radius_fin'' should be a scalar.")
                else:
                    sS['radius_fin']= content
                    isok=1
            elif field== 'alpha_ini':
                if not isinstance(content, np.array) or 'isscalar' not in locals():
                    print("'alpha_ini'' should be a scalar.")
                else:
                    sS['alpha_ini']= content
                    isok=1                    
            elif field == 'alpha_type':
                if not isinstance(content,str):
                    print("'alpha_type'' should be a string.")
                elif content != 'linear' and content != 'inv' and content != 'power' and content != 'constant' and content != '':
                    print("Unknown alpha type")
                    sS['alpha_type']= content
                    isok=1
                else:
                    sS['a;pha_type']= content
                    isok=1
            elif field== 'time':
                if not isinstance(content,str):
                    print("time should be a string")
                else:
                    sS['time']= content
                    isok=1 
            # if field is none of them:
            else:
                print(" Invalid field or data struct: ", field) 
            
        elif sS['type']== 'som_map':
            codebook = sS['codebook']
            if codebook.ndim==1:
                munits = 0
                dim = 0
            else:
                munits, dim = codebook.shape
                
            if field== 'codebook':
                if not isinstance(content, np.ndarray):
                    print("codebook should be a nurmeric matrix")
                elif content.shape != si and bool(sS['codebook']): 
                    print("'New ''codebook'' must be equal in size to the old one.'")
                elif 'iscomplete' not in locals():
                    print("Map codebook must not contains NaNs")
                else:
                    sS['codebook'] = content
                    isok=1
            elif field == 'labels':
                if  not bool(content):
                    sS['labels']= [[] for _ in range(munits)] 
                    isok=1
                elif len(content) != munits: 
                    print("Length of labels array must be equal to the number of map units.")
                elif not isinstance(content,list) and not isinstance(content,str): 
                    print("'labels'' must be a string array or a cell array/matrix.")
                else:
                    isok=1
                if isinstance(content,str):
                    content = [[c for c in s] for s in content]
                elif not (isinstance(content, list) and all(isinstance(item, str)) for item in content):
                    l = np.prod(content.shape)
                    for i in range(l):
                        if isinstance(content[i]):
                            if not bool(content[i]):
                                print("Invalid labels array")
                                isok=1
                            else:
                                content[i]=''
                elif isok==1:
                    sS['labels']= content      
                if isok ==1:
                    sS['labels']=content
            elif field== 'topol':
                if not isinstance(content,dict):
                    print("topol should be a topology dict")
                elif 'msize' not in content or 'lattice'  not in content or 'shape' not in content:
                    print("topol is not a valid topology dict")
                
                elif np.prod(content['msize']) != munits:
                    print("topol msize does not match the number of map units")
                else:
                    sS['topol']= content
                    isok=1
            elif field=='msize':
                if not isinstance(content, np.array) or 'isvector' not in locals() or 'ispositive' not in locals() or 'isinteger' not in locals():
                 	print("'msize'' should be a vector with positive integer elements.")
                elif np.prod(content) != munits:
                    print("msize'' does not match the map size.'")
                else:
                    sS['topol']['msize']= content
                    isok=1
            elif field== 'neigh':
                if not isinstance(content, str):
                    print(" neigh should be a string")
                elif content != 'gaussian' and content != 'ep' and content != 'cutgauss' and content != 'bubble':
                    print("unknown neighbourhood function")
                    sS['neigh']= content
                    isok=1
                else:
                    sS['neigh']=content
                    isok=1
            elif field== 'mask':
                if len(content) ==1:
                    content = [list(row) for row in zip(*content)]
                elif not isinstance(content, np.ndarray) or content.shape != (dim,1):
                    print("mask'' should be a column vector (size dim x 1).")
                else:
                    sS['mask']= content
                    isok=1
            elif field== 'comp_names':
                if not isinstance(content,list) and not isinstance(content,str):
                    print("comp_names'' should be a cell string or a string array.")
                elif len(content) != dim:
                    print("Length of ''comp_names'' should be equal to dim.")
                else:
                    if isinstance(content,str):
                        content = [[c for c in s] for s in content]
                    if len(content)==1:
                        content = [list(row) for row in zip(*content)]
                    sS['comp_names']= content
                    isok=1
            elif field=='name':
                if not isinstance(content, str):
                    print("'name'' should be a string.")
                else:
                    sS['name']= content
                    isok=1
                
            elif field == 'comp_norm': 
                if not isinstance(content,list) and len(content)>0:
                    print("comp_norm'' should be a cell array.")
                elif len(content) != dim:
                    print("Length of ''comp_norm'' should be equal to dim.")
                else:
                    isok=1
                for j, inner_list in enumerate(content):
                    if inner_list and (not isinstance(inner_list[0], dict) or 'type' not in inner_list[0] or inner_list[0]['type'] != 'som_norm'):
                        msg = "Each element in 'content' should be either empty or contain a dictionary of type 'som_norm'."
                        isok = False
                        break
                if isok==1:
                    sS['comp_norm']=content
                    isok=1
            elif field== 'trainhist':
                if not isinstance(content,dict) and bool(content):
                    print("'trainhist'' should be a struct array or empty.")
                else:
                    isok=1
                for key in content:
                    # Get the value associated with the current key
                    item = content[key]
    
                    # Check if item is a list and process accordingly
                    if isinstance(item, list):
                        for sub_item in item:
                            if sub_item and (not isinstance(sub_item, dict) or 'type' not in sub_item or sub_item['type'] != 'som_norm'):
                                msg = "Each element in 'comp_norm' should be either empty or type 'som_norm'."
                                isok = False
                                break
                        if isok !=1:
                            break
                        elif item and (not isinstance(item, dict) or 'type' not in item or item['type'] != 'som_norm'):
                            msg = "Each element in 'comp_norm' should be either empty or type 'som_norm'."
                            isok = False
                            break
                if isok==1:
                    sS['trainhist']=content
            
        elif sS['type']== 'som_topol':
            if field== 'msize':
                if not isinstance(content, np.ndarray) and 'isvector' not in locals() and 'ispositive' not in locals() and 'isinteger' not in locals():
                        print("msize'' should be a vector with positive integer elements.")
                else:
                    sS['msize']= content
                    isok=1
            elif field=='lattice':
                if not isinstance(content,str):
                    print("lattice'' should be a string'")
                elif content != 'rect' and content != 'hexa':
                    print("'Unknown lattice type: ' content")
                    sS['lattice'] = content
                    isok = 1
                else:
                    sS['lattice'] = content
                    isok = 1
            elif field== 'shape':
                if not isinstance(content,str):
                    print("shape'' should be a string")
                elif content !='sheet' and content != 'cyl' and content != 'toroid':
                    print("Unknown shape type: ' content") 
                    sS['shape'] = content; isok = 1;
                else:
                    sS['shape'] = content; isok = 1;
            else: 
                print("'Invalid field for topology struct: ' field")
    
                
                
                
                
        elif sS['type']== 'som_norm':
            if field== 'method':
                if not isinstance(field,str):
                    print("'method'' should be a string.")
                else:
                    sS['method'] = content
                    isok=1
            elif field == 'params':
                sS['params'] = content
                isok=1
            elif field=='status':
                if not isinstance(field,str):
                    print("'status' should be a string.")
                elif content != 'done' and content != 'undone' and content != 'uninit':
                    print("Unknown status type content")
                    sS['status']==content
                    isok=1
                else:
                    sS['status']==content
                    isok=1
            # if field is none of them:
            else:
                print(" Invalid field or data struct: ", field) 
        
        
        elif sS['type']=='som_data':
            temp= sS.get('data', np.array([]))
            shape = temp.shape
            if len(shape)==2:
                dlen, dim =temp.shape
            else:
                dlen, dim= 0,0
            
            if field=='data':
                dummy, dim2 = content.shape
                if np.prod(si)==0:
                    print(" 'data' is empty")
                elif not np.issubdtype(content.dtype, np.number):
                    print(" 'data'should be a nimeric matrix")
                elif dim != dim2 and len(sS['data'])!=0:
                    print("New data'' must have the same dimension as old one")
                else:
                    sS['data']=content
                    isok=1
            
            elif field == 'labels':
                if not bool(content):
                    sS['labels'] = [[] for _ in range(dlen)]
                    isok = 1;
                elif np.size(content,0) != dlen:
                     	print("Length of ''labels'' must be equal to the number of data vectors.")
                elif not isinstance(content, list) and not isinstance(content, str):
                    print(" labels must be a string array or a cell array/matrix.")
                else:
                    isok = 1 
                if isinstance(content,str):
                    content = [content]
                elif not (isinstance(content, list) and all(isinstance(elem, str) for elem in content)):
                    l =np.prod(np.size(content));
                    for j in range(l): 
                          if not isinstance(content[j],str): 
                              if bool(content[j]): 
                                  print(" Invalid 'labels' array.")
                                  k = 0
                              else:
                                  content[j] = ''
                if isok==1:
                    sS['labels'] = content
            
            elif field=='name':
                if not isinstance(content,str):
                    print(" 'name'should be a string")
                else:
                    sS['name']=content
                    isok=1
                    
            elif field == 'comp_names':
                if not isinstance(content, (list, str)):
                    print(" 'comp_names' should be a cell string or a string array")
                elif np.size(content) != dim:
                    print(" Length of 'comp_names'should be equal to dim")
                else:
                    if isinstance(content, str):
                        content = [content]
                    if np.size(content)==1:
                        content = np.transpose(content)
                    sS['comp_names'] = content
                    isok=1
                    
            elif field == 'comp_norm':
                if not isinstance(content, (list)) and len(content)>0:
                    print("'comp_norm'' should be a cell array.")
                elif len(content) !=dim:
                    print("Length of ''comp_norm'' should be equal to dim.")
                else:
                    isok = 1
                    sS['comp_norm']=content
                 
                 
            elif field == 'label_names':
                if not isinstance(content, (list)) and not isinstance(content, (str)) and bool(content):
                    print("'label_names'' should be a cell string, a string array or' empty.")
                else:
                    if bool(content):
                        if isinstance(content, (str)):
                            content = [[c for c in s] for s in content]
                        if len(content)==1:
                            content = [list(row) for row in zip(*content)]
                sS['label_names']=content
                isok = 1
              
            # if field is none of them:
            else:
                print(" Invalid field or data struct: ", field) 
                    
              
    return sS
                
                

