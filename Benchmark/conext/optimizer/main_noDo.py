# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:45:09 2022

@author: antonio
"""

from __future__ import print_function

import sys, os

if os.path.join('..','helpers') not in sys.path:
    sys.path.append(os.path.join('..','helpers')) # os.getcwd()

from subprocess import call
import re
import numpy as np
import time
from natsort import natsorted
import glob

import pandas as pd

import networkx as nx
import solver
import benders
from helpers import *


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

def load_sets(Fpath_Dataset):
    """
    Loads the data from the given file path.
    """    
    list_to_df = lambda x: pd.DataFrame(x).T
    
    SP, OB, TRAF, SP_names, OB_names, TRAF_names = [], [], [], [], [], []
    
    typee=['*SP_120min.npy','*OB_30min.npy','*traff.npy'] 
    
    load_data(Fpath_Dataset, typee[0], output_list=SP, services_name_list=SP_names, printt=False)
    load_data(Fpath_Dataset, typee[1], output_list=OB, services_name_list=OB_names ,printt=False)
    load_data(Fpath_Dataset, typee[2], output_list=TRAF,services_name_list=TRAF_names, printt=False)
    SP=list_to_df(SP)
    OB=list_to_df(OB)
    TRAF=list_to_df(TRAF)

    return SP, OB, TRAF, get_sliceNnames(TRAF_names)

###############################################################################
#%% LOADING DATASETS
###############################################################################
print("---- Loading data -----") 
Filepath= os.path.join('..','Paris','')

SP_db, OB_db, TRAF_db, Slice_names = load_sets(Filepath)

print("---- data loaded -----") 

###############################################################################
#%% PARAMETERS
###############################################################################

C_tot = 10
SliceDuration = 10

delayToRemove = 5

SP   = SP_db[:SliceDuration]
OB   = OB_db[:SliceDuration]
TRAF = TRAF_db[:SliceDuration]
    
tenant_rewards = SP.iloc[0,ii] # p_s in INFOCOM22
 
leasing_cost = 100

mypredictor = 'holtwinters'

monitoring_samples = 30  # Monitoring samples in one decision interval
n_preds            = 4
slen               = 24*60/monitoring_samples # season num of samples


###############################################################################
#%% Conext code
###############################################################################
print("############# Compute Optimal Allocation ####################")

## 0) Read data from json file (TBD: Received from socket?)
#0.1) Topology

G = nx.Graph()
G.add_nodes_from([1,2])
G.add_edge(1,2)
# print(G)

mynetwork = Network(G) 
base_station = [BaseStation(1, C_tot)]
dc = [DC(1, C_tot)]
myinfrastructure = Infrastructure(mynetwork, base_station, dc)

myinfrastructure.leasing_cost = leasing_cost

allp = k_shortest_paths(myinfrastructure.network.graph,
                        myinfrastructure.base_stations[0].idx, 
                        myinfrastructure.dcs[0].idx, k=1)

myinfrastructure.paths['1'] = {}
myinfrastructure.paths['1']['1'] = {}
myinfrastructure.paths['1']['1']['path'] = allp[1]
myinfrastructure.paths['1']['1']['latency'] = [None]

NumBS = len(myinfrastructure.base_stations)

### Setting tenants
mytenants = []
for ii in range(len(TRAF)):
    # SP remains constant for the block, then we get only one value
    myrequest = Request(bitrate = {'1':SP.iloc[0,ii]}, delay = delayToRemove, duration = SliceDuration)
            
    tenant = Tenant(idx=ii, 
                    penalty = 0,
                    reward = tenant_rewards[ii],
                    request = myrequest,
                    # forecast_usage = {},
                    # forecast_uncertainty = {},
                    # accepted = 0,
                    # nw_alloc = {},
                    # path_alloc = {},
                    # dc_alloc = [],
                    # data_history = {},
                    # last_accepted = -1,
                    # service = {},
                    # status = "PENDING", #ACCEPTED, REJECTED, OFFLINE
                    type_tenant = None)
    
    mytenants.append(tenant)
    
NumTenants = len(mytenants)


#%%############################################################################
###############################################################################
print("############# Tenant Traffic Predictor ####################")
###############################################################################
###############################################################################
  
if True:
    if mypredictor == "lstm":
        import tenant_predictor_lstm as predictor
    elif mypredictor == "holtwinters":
        import tenant_predictor_holtwinters as predictor
    else:
        raise Exception("ERRROR: Incorrect predictor name")

    #0.2) Tenant requests
    # alltenants = AllTenants()
    # alltenants.fromJSON(tenants_file)


    #Generate  FORECAST
    for tau in range(0,NumTenants):
        tenantTau = mytenants[tau]
        print("Tenant {0}: ".format(tenantTau.idx))
        
        idx_BS_str = str(myinfrastructure.base_stations[0].idxs)
        
        # for i in range( 0 ,NumBS):
        TrainingTraffic = [tenantTau.data_history[idx_BS_str][str(k)] for k in range(len(tenantTau.data_history[idx_BS_str])) ] # get history data in order
        
        myrequest = float(tenantTau.request.bitrate[idx_BS_str])
        
        forecast, forecast_uncertainty = predictor.make_forecast(TrainingTraffic, myrequest, tenantTau.idx, idx_BS_str, monitoring_samples);   
                
        tenantTau.forecast_usage[idx_BS_str] = min(forecast, float(tenantTau.request.bitrate[idx_BS_str]))
        tenantTau.forecast_uncertainty[idx_BS_str] = min(forecast_uncertainty,  float(tenantTau.request.bitrate[idx_BS_str]))            

        print("BS {0}, forecast = {1} + {2} Mb/s".format(idx_BS, tenantTau.forecast_usage[idx_BS_str], tenantTau.forecast_uncertainty[idx_BS_str])) 
            
    # Write tenants file        
    for tau in range(0, NumTenants):
        for i in range( 0, NumBS):
            if len(mytenants[tau].data_history[str(idx_BS)]) < mon_samples: # clean it...            
                mytenants[tau].data_history[str(idx_BS)] = []
                    
    alltenants.toJSON(tenants_file)

    


#0.2) Tenant requests PRINTING
for tau in range(len(mytenants)):
    thisTenant = mytenants[tau]
    bsIdx      = myinfrastructure.base_stations[0].idx
    print("Tenant {0}: ".format(mytenants[tau].idx))
    print("All BSs, request={:.2f},".format (thisTenant.request.bitrate[str(bsIdx)]),
          "reward={}, penalty={}, duration={}".format( 
        thisTenant.reward, thisTenant.penalty, thisTenant.request.duration_remaining))

# for i in range(len(myinfrastructure.base_stations)):
# for d in range(len(myinfrastructure.dcs)):
        # print(f"Paths from: BS {i} to CU {d}") 
for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[0].idx)][str(myinfrastructure.dcs[0].idx)]['path'])):
    print("    Path:", myinfrastructure.paths[str(myinfrastructure.base_stations[0].idx)][str(myinfrastructure.dcs[0].idx)]['path'][(p)])
    print("    Path latency:", myinfrastructure.paths[str(myinfrastructure.base_stations[0].idx)][str(myinfrastructure.dcs[0].idx)]['latency'][(p)])



###############################################################################
#%% Solve it ##
###############################################################################
## 3) 
zsol, xsol, usol, vsol, wsol, myindex = solver.solve(mytenants, myinfrastructure)


###############################################################################
#%% Printing solution
###############################################################################print("############# SOLUTION ####################")

rejected_list = []
accepted_list = []
tau_online = 0
n_embb_accepted = 0
n_mmtc_accepted = 0
n_urlcc_accepted = 0
for tau in range(len(mytenants)):
    if mytenants[tau].idx in list_online_tenants_idx:
        rejected = 1
        #print("Tenant {0}: ".format(mytenants[tau].idx))
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):
                    #print(tau_online)
                    if zsol[myindex[str(tau_online) + "." + str(i) + "." + str(d) + "." + str(p)]] > 1e-6 or xsol[myindex[str(tau_online) + "." + str(i) + "." + str(d) + "." + str(p)]] > 0.5:
                        print("[tau {0}] BS {1}, DC {2}, path {3}, bitrate = {4:0.01f} Mb/s, x={5} (forecast={6:0.01f} + {7:0.01f} Mb/s, Reward={7}, Penalty={8})".format(tau, i, d, myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p], zsol[myindex[str(tau_online) + "." + str(i) + "." + str(d) + "." + str(p)]], xsol[myindex[str(tau_online) + "." + str(i) + "." + str(d) + "." + str(p)]], mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)], mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[i].idx)], float(mytenants[tau].reward), float(mytenants[tau].penalty)))
                        mytenants[tau].nw_alloc[str(myinfrastructure.base_stations[i].idx)] = zsol[myindex[str(tau_online) + "." + str(i) + "." + str(d) + "." + str(p)]]
                        mytenants[tau].dc_alloc = myinfrastructure.dcs[d].idx
                        mytenants[tau].path_alloc[str(myinfrastructure.base_stations[i].idx)] = myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p]
                        rejected = 0
        if rejected == 1:
            rejected_list.append(mytenants[tau].idx)
            mytenants[tau].accepted = 0
            mytenants[tau].status = "REJECTED"
        else:
            accepted_list.append(mytenants[tau].idx)
            mytenants[tau].accepted = 1
            if mytenants[tau].type_tenant == 1:
                n_embb_accepted = n_embb_accepted + 1
            elif mytenants[tau].type_tenant == 2:
                n_mmtc_accepted = n_mmtc_accepted + 1
            elif mytenants[tau].type_tenant == 3:
                n_urlcc_accepted = n_urlcc_accepted + 1    
            mytenants[tau].status = "ACCEPTED"
            mytenants[tau].last_accepted = alltenants.decision_interval
            #if int(mytenants[tau].request.duration_remaining) > 0:
                #mytenants[tau].request.duration_remaining = int(mytenants[tau].request.duration_remaining) - 1
            #if  mytenants[tau].request.duration_remaining == 0: 

                #if int(mytenants[tau].request.loop) == 1:
                    #mytenants[tau].request.duration_remaining = mytenants[tau].request.duration
                    #mytenants[tau].status = "PENDING"
                    #mytenants[tau].accepted = 0
                #else:                            
                    #mytenants[tau].status = "OFFLINE"
                    #mytenants[tau].accepted = 0
                #mytenants[tau].request.duration = 1
        tau_online = tau_online + 1
    
print("Tenants rejected: {0}".format(rejected_list))
print("Tenants accepted: {0}".format(accepted_list))

#f = open(ResultFile, 'a')
#f.write(str(n_embb_accepted) + "\t" + str(n_mmtc_accepted) + "\t" + str(n_urlcc_accepted)  + "\t")
#f.close()        

for d in range(len(myinfrastructure.dcs)):
    #print("usol={}".format(usol[d]))
    if usol[d] > 1e-2:
        print("CU {0} leased capacity={1:0.01f} Mb/s".format(d, usol[d]))
        #variable = raw_input('Please sth to continue')        

for i in range(len(myinfrastructure.base_stations)):
    #print("vsol={}".format(vsol[i]))
    if vsol[i] > 1e-2:
        print("BS {0} leased capacity={1:0.01f} Mb/s".format(i, vsol[i]))
        #variable = raw_input('Please sth to continue')

nlinks = myinfrastructure.network.graph.number_of_edges()
linkid=0
for link_ in myinfrastructure.network.graph.edges_iter(data=True):
    if wsol[i] > 1e-2:
        print("link {0} leased capacity={1:0.01f} Mb/s".format(link_, wsol[i]))
        #variable = raw_input('Please sth to continue')
    linkid = linkid + 1


    
#update interval counter and write to json file    
alltenants.decision_interval = alltenants.decision_interval + 1
alltenants.toJSON(tenants_file)