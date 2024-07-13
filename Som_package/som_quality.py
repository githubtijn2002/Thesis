# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:25:10 2024

@author: sdd380
"""
import numpy as np
from som_bmus import som_bmus
from som_unit_neighs import som_unit_neighs
from scipy.sparse import csr_matrix

def som_quality(sMap, D):
    # DESCRIPTION
    #
    # This function measures the quality of the given map. The measures are
    # data-dependent: they measure the map in terms of the given
    # data. Typically, the quality of the map is measured in terms of the
    # training data. The returned quality measures are average quantization
    # error and topographic error.
    #
    # The issue of SOM quality is a complicated one. Typically two evaluation
    # criterias are used: resolution and topology preservation. There are
    # many ways to measure them. The ones implemented here were chosen for
    # their simplicity.
    # qe : Average distance between each data vector and its BMU.
    #       Measures map resolution.
    #  te : Topographic error, the proportion of all data vectors
    #       for which first and second BMUs are not adjacent units.
    #       Measures topology preservation.
    # REFERENCES
    #
    # Kohonen, T., "Self-Organizing Map", 2nd ed., Springer-Verlag, 
    #    Berlin, 1995, pp. 113.
    # Kiviluoto, K., "Topology Preservation in Self-Organizing Maps", 
    #    in the proceeding of International Conference on Neural
    #    Networks (ICNN), 1996, pp. 294-299.
    
    # check arguments
    if sMap is None or D is None:
       raise TypeError("Not enough input arguments.")
    
    if isinstance(D, dict):
        D = D['data']
    dlen, dim = D.shape
    
    b = np.arange(0,2)
    bmus, qerrs = som_bmus(sMap, D, b)
    
    inds = np.where(~np.isnan(bmus[:,0]))[0]
    bmus = bmus[inds,:]
    qerrs = qerrs[inds,:]
    l = len(inds)
    if l==0:
        raise ValueError('Empty data set')
    
    # mean quantization error
    mqe = np.mean(qerrs[:,0])
    
    if len(b)==2:
        Ne1= som_unit_neighs(sMap['topol'])
        Ne = Ne1.toarray()
        
        tge = 0
        # breakpoint()
        for i in range(l):
            row_idx = int(bmus[i, 0])
            col_idx = int(bmus[i, 1])
        
            # Check if indices are within bounds of Ne
            if 0 <= row_idx < Ne.shape[0] and 0 <= col_idx < Ne.shape[1]:
                # Check if Ne[row_idx, col_idx] is not equal to 1
                if int(Ne[row_idx, col_idx]) != 1:
                    tge += 1
    
        tge = tge / l
    else:
        tge = np.nan

    
    return mqe , tge