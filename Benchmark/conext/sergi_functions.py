# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:00:12 2022

@author: antonio
"""
import os,sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from natsort import natsorted
import json

sys.path.append('./dashboard/')
sys.path.append('./orchestrator/')


###############################################################################
#%% Load Data Functions
###############################################################################
def load_data(Data_Fpath,typee,output_list,services_name_list,printt=False):
        for f in natsorted(glob.glob(Data_Fpath+typee)):
        
            output_list.append(np.load(f))
            
            split_linux   = f.split('/')[-1].split('.')[0]
            split_windows = split_linux.split('\\')[-1].split('.')[0]
            services_name_list.append(split_windows)
            if printt == True:
                print(f)
                print(split_windows)
            if not output_list:
                print ('Load Error, check Filepath Data :'+str(typee))
                
def get_sliceNnames(slices):
    Slice_list=[]
    for i in range(len(slices)):
        Slice_list.append(slices[i].split('_traff')[0])
    
    if '_' in slices[0].split('_traff')[0]:
        Slice_list=[]
        for i in range(len(slices)):
             Slice_list.append(slices[i].split('_traff')[0].split(str(i)+'_')[-1])

    return Slice_list

def list_to_df(listt):
        return pd.DataFrame(listt).T
    
def load_sets(Fpath_Dataset, val = 0,synthetic_dataset=0):
    """
    Loads the data from the given file path.
    """
    SP, TRAF, SP_names, TRAF_names = [], [], [], []
    if val == 0:
        if  Fpath_Dataset.__contains__('forecast'):
            typee=['*120min_prediction.npy','*clean.npy'] 
            if synthetic_dataset == 1:
                typee=['*120min_prediction.npy','*agg_60_s.npy'] 
        else:
            typee=['*SP_120min.npy','*traff.npy'] 
            # sys.exit('unknown folder, please check load_set() function.')   
            
        load_data(Fpath_Dataset,typee[0],output_list=SP,services_name_list=SP_names,printt=False)
        load_data(Fpath_Dataset,typee[1],output_list=TRAF,services_name_list=TRAF_names,printt=False)
        
        SP=list_to_df(SP)
        TRAF=list_to_df(TRAF)
        
        return SP, TRAF #, get_sliceNnames(TRAF_names)
    else:
        if synthetic_dataset == 1:            
            load_data(Fpath_Dataset,'*agg_60_s.npy',output_list=TRAF,services_name_list=TRAF_names,printt=False)     
        else:
            load_data(Fpath_Dataset,'*clean.npy',output_list=TRAF,services_name_list=TRAF_names,printt=False)  

        TRAF=list_to_df(TRAF)
        return TRAF

    

