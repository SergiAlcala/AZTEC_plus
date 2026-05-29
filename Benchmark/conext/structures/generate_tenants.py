from __future__ import print_function

import sys
sys.path.append('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared/helpers/')
sys.path.append('../')
import os

from subprocess import call
import re
import numpy as np
import time
import json
import pandas as pd
from tqdm import tqdm


from helpers import *
import sergi_functions 

# sys.path.append('./dashboard/')
# sys.path.append(os.path.join('.','structures'))
sys.path.append('C:\\Users\\anton\\ownCloud\\Sergi\\Overbooking\\2022_07_benchmark\\structures\\')

def generateTenant(allTenantsObj, req_traf, real_traf_past, real_traf_next, json2write, t_id, T_slice_mno, reward, topo_file):
    print(f"#### Generate Tenant Request {t_id} ####") # for this interval
    ## Tenant requests allTenantsObj this time ##
    # mytenants = AllTenants()
    mytenants = allTenantsObj
    myinfrastructure = Infrastructure(Network(nx.Graph()) , [], [])
    myinfrastructure.fromJSON(topo_file)  
    
    type_tenants  = 0 # deprecated
    
    delay         = 1e3
    service_mean  = req_traf
    service_std   = req_traf          
    service_alpha = 0
    service_beta  = 0 #2, 0.2, 0 
    penalty       = 2*reward         

    mybitrate = {}
    mydata_history = {}
    mydata_future = {}
    myservice_mean = {}
    myservice_std = {}    
    for b in range(len(myinfrastructure.base_stations)):
        mybitrate[str(myinfrastructure.base_stations[b].idx)] = req_traf
        mydata_history[str(myinfrastructure.base_stations[b].idx)] = dict(zip([str(e) for e in range(len(real_traf_past))], real_traf_past))
        mydata_future[str(myinfrastructure.base_stations[b].idx)] = dict(zip([str(e) for e in  range(len(real_traf_next))], real_traf_next))
        myservice_mean[str(myinfrastructure.base_stations[b].idx)] = service_mean
        myservice_std[str(myinfrastructure.base_stations[b].idx)] = service_std
        
    myrequest = Request(mybitrate, delay, T_slice_mno)
    
    tenant = Tenant(t_id, penalty, reward, myrequest, type_tenants) # idx, penalty, reward, request
    tenant.service = Service(myservice_mean, myservice_std, service_alpha, service_beta)
    tenant.data_history = mydata_history
    tenant.data_future  = mydata_future
    tenant.request.loop = 0
    mytenants.tenant.append(tenant)

    for tau in range(len(mytenants.tenant)):
        print("Tenant {0} (delay={1}): ".format(mytenants.tenant[tau].idx, mytenants.tenant[tau].request.delay))
        print("BSs, bitrate request = {0} Mb/s".format(mytenants.tenant[tau].request.bitrate[str(myinfrastructure.base_stations[0].idx)])) 

    # write tenants file
    mytenants.toJSON(json2write)    

def generate_Paris_tenants(n_tenants, reqBitrate, trafHist, trafNext, T_slice_mno, 
                           reward, json2write, topo_file):
    mytenants = AllTenants()   
    for ii in range(n_tenants): # num services
        generateTenant(mytenants, reqBitrate[ii], trafHist[ii], trafNext[ii], json2write, ii, T_slice_mno, reward[ii], topo_file)
    return mytenants
        
#%%

def createTenantFiles(Filepath_test, Filepath_val, nTenants, lengthHistory, start_test,
                      monetary_factor, T_Slice, T_mno, idx_last, topo_file, json_dir,synthetic_dataset):
    print(f"############## Loading datasets from npy files.... #############")
    ## LOADING DATASETS
    req_TRAFt, real_TRAFt = sergi_functions.load_sets(Filepath_test,synthetic_dataset=synthetic_dataset)
    real_TRAFv            = sergi_functions.load_sets(Filepath_val, val = 1,synthetic_dataset=synthetic_dataset)

    print(f"############## Generating samples... ##############")
    print(f'Requested traf shape: {req_TRAFt.shape}')
    print(f'Real traf shape: {real_TRAFt.shape}')
    print(f'Real traf val shape: {real_TRAFv.shape}')
    
    ## Generating samples...
    # json2write = os.path.join('..','structures','tenants.json')
    print(topo_file)
    T_experiment = idx_last+1
    
    SP_test       = req_TRAFt.iloc [ : idx_last, :]
    TRAF_test     = real_TRAFt.iloc[ : idx_last, :]
    TRAF_val_hist = real_TRAFv.iloc[start_test - lengthHistory: start_test, :]

    
    traf_merged_history = pd.concat([TRAF_val_hist,TRAF_test], axis=0)
    
    num_T_slice = int(np.floor(T_experiment/T_Slice))
        
    for tt in tqdm(range(num_T_slice)):
        print(f"############## Generate Tenants Requests (for decision interval {tt}) ##############")
        dec_block   = T_Slice*tt #T_mno
        req_bitrate = SP_test.iloc[dec_block,:]
        traf_hist   = [traf_merged_history.iloc[dec_block : dec_block + lengthHistory, ii].tolist() for ii in TRAF_test.columns]
        traf_next   = [traf_merged_history.iloc[dec_block + lengthHistory : dec_block + lengthHistory + T_Slice, ii].tolist() for ii in TRAF_test.columns]
        reward      = monetary_factor*req_bitrate
    
        json2write = os.path.join(json_dir,f'tenants_{tt}.json')
        # print(json2write)
        
        alltenants = generate_Paris_tenants(nTenants, req_bitrate, traf_hist, traf_next, 
                                            int(T_Slice/T_mno), reward, json2write, topo_file)
    
    myinfrastructure = Infrastructure(Network(nx.Graph()) , [], [])
    myinfrastructure.fromJSON(topo_file)  
    for ten in alltenants.tenant:
        print("Tenant {0} (delay={1}): ".format(ten.idx, ten.request.delay))
        print("BSs, bitrate request = {0} Mb/s".format(ten.request.bitrate[str(myinfrastructure.base_stations[0].idx)])) 
    # print("############# Creating tenant files for all TS... ####################")

    return TRAF_test, SP_test
    
