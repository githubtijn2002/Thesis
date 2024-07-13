# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:48:54 2024

@author: sdd380
"""
import numpy as np


def som_bmus(sMap, sData, which_bmus, *mask):
    #PURPOSE
    #
    # Finds Best-Matching Units (BMUs) for given data vector from a given map.
    ## Returns the indexes and corresponding quantization errors of the
    # vectors in sMap that best matched the vectors in sData.
    #
    # By default only the index of the best matching unit (/vector) is
    # returned, but the 'which' argument can be used to get others as
    # well. For example it might be desirable to get also second- and
    # third-best matching units as well (which = [1:3]). 
    #
    # A mask can be used to weight the search process. The mask is used to
    # weight the influence of components in the distance calculation, as
    # follows: 
    #
    #   distance(x,y) = (x-y)' diag(mask) (x-y)
    #
    # where x and y are two vectors, and diag(mask) is a diagonal matrix with 
    # the elements of mask vector on the diagonal. 
    #
    # The vectors in the data set (sData) can contain unknown components
    # (NaNs), but the map (sMap) cannot. If there are completely empty
    # vectors (all NaNs), the returned BMUs and quantization errors for those 
    # vectors are NaNs.
    
    # check arguments
    # tbd
    
    # sMap
    if isinstance(sMap, dict):
        if sMap['type']=='som_map':
            M = sMap['codebook']
        elif sMap['type'] == 'som_data':
            M= sMap['data']
        else:
            raise ValueError('Invalid 1st argument')
    else:
        M = sMap
    munits, dim = M.shape
    if np.any(np.any(np.isnan(M))):
        raise ValueError('Map codebook must not have missing components.')
    
    # data
    if isinstance(sData, dict):
        if sData['type']=='som_map':
            D = sData['codebook']
        elif sData['type']=='som_data':
            D = sData['data']
        else:
            raise ValueError('Invalid 2nd argument.')
    else:
        D = sData
    dlen, ddim = D.shape
    
    if dim != ddim:
        raise ValueError('Data and map dimensions do not match.')
    
    # which bmus
    if sMap is None or sData is None or which_bmus is None or len(which_bmus)==0:
        which_bmus=1
    else:
        if isinstance(which_bmus, str):
            if which_bmus== 'best':
                which_bmus=[1]
            elif which_bmus== 'worst':
                which_bmus= munits
            elif which_bmus== 'all':
                which_bmus= list(range(0,munits))

    # mask
    if sMap is None or sData is None or which_bmus is None or not bool(mask) or np.any(np.isnan(mask)):
        if isinstance(sMap,dict) and sMap['type']=='som_map':
            mask = sMap['mask']
        elif isinstance(sData, dict) and sData['type']=='som_map':
            mask = sData['mask']
        else:
            mask = np.ones((dim,1))
    
    if len(mask)==1:
        mask = mask.T
    if np.all(mask)==0:
        raise ValueError('All components masked off. BMU search cannot be done.')
    
    ## ACTION
    Bmus = np.zeros((dlen, len(which_bmus)))
    Qerrors = Bmus.copy()
    
    blen = min((munits,dlen))
    
    Known =  (~np.isnan(D)).astype(int)
    W1 = (mask * np.ones((1, dlen)) * Known.T)
    D[~Known.astype(bool)] = 0
    unknown = np.where(sum(Known.T)==0)
    
    # constat matrices
    WD = 2 * np.dot(np.diag(mask.flatten()), D.T)
    dconst = np.dot((D**2),mask)
    
    i0=0
    while i0 + 1 <= dlen:
      inds = np.arange(i0, min(dlen, i0 + blen))
      i0 += blen
      Dist = np.square(M) @ W1[:, inds] - M @ WD[:, inds]
      # find the bmus and the corresponding quantization errors
      if np.all(which_bmus==1):
          Q = np.min(Dist, axis=0)
          B = np.argmin(Dist, axis=0)
      else:
          Q = np.sort(Dist, axis=0)
          B = np.argsort(Dist, axis=0)

      if munits ==1:
          Bmus[inds,:]=1
      else:
          Bmus[inds,:]= B[which_bmus,:].T
      Qerrors[inds,:] = Q[which_bmus,:].T + dconst[inds, np.ones((len(which_bmus),1), dtype=int) - 1].T
      
    #completely unknown vectors
    if len(unknown)>1:
        Bmus[unknown,:] = np.nan
        Qerrors[unknown,:] = np.nan
    
    Qerrors = np.sqrt(Qerrors)
          

    
    return Bmus, Qerrors