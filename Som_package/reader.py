# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:22:24 2024

@author: sdd380
"""
import csv
import numpy as np

def read_headers(filename):
    with open(filename, 'r') as f:
        headers = f.readline().strip().split(',')
    return headers

def read_data(filename):
    return np.loadtxt(filename, delimiter=',', skiprows=1)