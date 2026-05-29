# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 09:02:26 2022

@author: antonio
"""
#!/usr/bin/python
import sys, os, re
import numpy as np
import math
import time
import json
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import date
from multiprocessing import Pool


from networkx.readwrite import json_graph
import networkx as nx

sys.path.append('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared/dashboard')
sys.path.append('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared/orchestrator/')
sys.path.append('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared/optimizer/')
sys.path.append('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared/helpers/')
sys.path.append('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared/structures/')

# import generate_topology
from generate_topology import generate_onelink_topology
import generate_tenants
import orchestrator
import tenant_predictor
from helpers import *
import main

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))
def clean():
    print("Cleaning up...")
    # purge("/tmp/", "enso_tenants_state.json")
    # purge("/tmp/", "enso_tenants_dashboard.json")    

def usage():
    print("""\
        Usage:
run(toponame, n_tenants, penalty, type (a value per tenant), lambda (a value per tenant between 0 and 1), sigma (a value per tenant between 0 and 1)
type 1: eMBB (LAMBDA = 50, latency = 20ms, beta = 0, R = 1)
type 2: mMTC (LAMBDA = 10, sigma = 0, latency = 20ms, beta = 1, R = 1)
type 3: uRLCC (LAMBDA = 25, latency = 5ms, beta = 0.25, R = 1)
Exiting...""")
# run()
################################################################################
#%%
################################################################################
#T_mno_list                = [5,15,30,60,120]


#T_mno                = 15 #[5,15,30,60,120]



Ctots=[0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]

today_date = date.today().strftime("%Y_%m_%d")

def myMultiOpt(idx):

    synthetic_dataset,Ctot,T_mno,doFiles=idx

    print('You have selected', T_mno, 'allocation')

    T_Slice_interval_run = int(9480/T_mno) #316

    
    if synthetic_dataset == 1:

        path_sourceNPY_t = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','syn_data_noisy_108_forecast_Norm','')
        path_sourceNPY_v = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','syn_data_noisy_108_long_Norm','')
        folder_results = f'T_ADM/{today_date}_{T_mno}min_2daysHist_Synthetic_Data'
        print('You have selected the synthetic dataset')

    else:
        path_sourceNPY_t = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','Paris_forecast','')
        path_sourceNPY_v = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','Paris_long','')
        folder_results = f'{today_date}_{T_mno}min_2daysHist_Real_Data'
        print('You have selected the real dataset')


    sourceTenantDir       = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', folder_results)

    ### Directory and file creation ##############################

    folder_results_path = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','results', folder_results)
    ii = 0
    while os.path.isdir(folder_results_path) :
        folder_results_path = folder_results_path[:-1] + str(ii)
        ii += 1
    os.mkdir(os.path.join('.', folder_results_path))

    ResultFile            = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', folder_results_path,'tenant_results.dat')
    tenant_dashboard_file = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', 'structures', 'tenant_dashboard.json')
    tenants_file          = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', folder_results_path,'tenant_file.json')
    
    json_dir              = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', 'structures')
    file_pkl_traf         = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', 'structures', 'source_traffic.pkl')
    file_pkl_x_z          = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', folder_results_path, 'results.pkl')
    empty_file            = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', 'structures', 'empty_file.json')

    #path_sourceNPY_t = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','Paris_forecast','')
    #path_sourceNPY_v = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','Paris_long','')
    #path_sourceNPY_t = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','syn_data_forecast','')
    #path_sourceNPY_v = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared','syn_data_long_Norm','')
    ### parameters ################################################################






    
    #doFiles    = 0 # If the filesof tenants per decision interval are already in folder "structures", then Fale
    doTopology = 1 # If you want to renovate and update the topology

    nTenants        = 20
    monetary_factor = 1

    mon_samples     = 24*60*2
    T_Slice         = 120
    dur_slice       = T_Slice/T_mno
    lengthHistory   = mon_samples

    train_samples, val_samples = 40320, 20160 #, htest_samples  = 10080 - 480 - 120
    start_test    = train_samples + val_samples + 480

    idx_first     = 0
    idx_last      = idx_first + T_Slice_interval_run*T_mno 

    leasing_cost = 1

    #### if needed, generate JSON files  ###########################################
    
    topo_file             = os.path.join('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared', 'structures', f'oneLink_topo_.json')
    if doTopology:
        print(f'Generating topology file for {Ctot}')
        generate_onelink_topology(topo_file,capacity=Ctot)


    if doFiles:
        savepath=(os.path.join('/home/jupyter-sergi/2023_02_04 - CoNEXT benchmark - shared', sourceTenantDir))
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        realTraf, reqTraf = generate_tenants.createTenantFiles(path_sourceNPY_t, path_sourceNPY_v, 
                            nTenants, lengthHistory, start_test, monetary_factor, T_Slice,
                            T_mno, idx_last, topo_file, sourceTenantDir,synthetic_dataset)

        print(f' Real Traff Len: {len(realTraf)}')
        print(f' Req Traff Len: {len(reqTraf)}')

        with open(file_pkl_traf, 'wb')  as pkl_traf:
            pickle.dump(realTraf, pkl_traf)
            pickle.dump(reqTraf, pkl_traf)
        sys.exit(f'All tenants generated in folder. , Real_Traf_len: {len(realTraf)}, Req_Traf_len: {len(reqTraf)}, T_mno: {T_mno}, C_tot: {Ctot}')
        # print(f'All tenants generated in folder. , Real_Traf_len: {len(realTraf)}, Req_Traf_len: {len(reqTraf)}, T_mno: {T_mno}, C_tot: {Ctot}')
    else:
        with open(file_pkl_traf, 'rb')  as pkl_traf:
            realTraf = pickle.load( pkl_traf)
            reqTraf = pickle.load( pkl_traf)   
            
        
   

    ################################################################################
    #%% simulation #################################################################
    ################################################################################

    total_epochs = int((idx_last - idx_first + 1)/T_mno)

    z_sol = np.zeros((total_epochs, nTenants))
    x_sol = np.zeros((total_epochs, nTenants))

    # total_epochs = 16
    n_epoch = 0
    for n_epoch in tqdm(range(total_epochs)):
    # if True:
        print("########################################################################\n",
            f"#################################### STARTING EPOCH {n_epoch}... ####################################\n",
            "########################################################################\n")
        f = open(ResultFile, 'a')
        f.write(str(n_epoch) + "\t")
        f.close()
        
        if n_epoch % dur_slice == 0:
            tenant_dashboard_file = os.path.join(sourceTenantDir,f'tenants_{int(n_epoch/dur_slice)}.json')
            arrivalTime = True
            if n_epoch != 0:
                os.remove(tenants_file) 
        else:
            arrivalTime = False
            
            
            # 1) Retrieve data from dashboard
        orchestrator.sync_orchestrator(topo_file, tenant_dashboard_file, tenants_file, arrivalTime)
        
        # 2) Make predictions
        predictor = "holtwinters" # lstm/holtwinters
        tenant_predictor.do(predictor, topo_file, tenants_file, T_mno, mon_samples)
        
            
    #     # 3) Find solution
    #     z_sol[n_epoch,:], x_sol[n_epoch,:] = main.do(topo_file, tenants_file, leasing_cost)
        
    #     #4) Sync tenant data...
    #     # orchestrator.sync_dashboard(tenants_file, tenant_dashboard_file) 

    #     #5) Orchestrator provides history data per tenant and BS 
    #     # orchestrator.get_tenant_samples() TBD: get monitoring data from from orchestrator
    #     orchestrator.update_tenant_samples(topo_file, tenants_file, ResultFile, T_mno)     
        
    #     #raw_input("Press key to continue...")
    #     #raw_input("Press key to continue...")


    # xxx = x_sol.copy()
    # zzz = z_sol.copy()

    # with open(file_pkl_x_z, 'wb')  as pkl_res:
    #     pickle.dump(xxx, pkl_res)
    #     pickle.dump(zzz, pkl_res)

    # with open(file_pkl_sol, 'rb')  as pkl_sol:
    #     x = pickle.load( pkl_sol)
    #     z = pickle.load( pkl_sol)   
            
    # #%%
    # import pandas as pd
    # reserved_traff =  np.sum(x_sol*z_sol, axis = 1)
    # # res_traff_mon  =  pd.Series(np.kron(reserved_traff, np.ones(T_mno)))
    # res_traff_mon  =  np.kron(reserved_traff, np.ones(T_mno))

    # realTraf_sum   = np.sum(realTraf, axis=1)
    # reqTraf_sum    = np.sum(reqTraf, axis=1)

    # plt.plot(reqTraf_sum, label='Requested traff', linewidth=.8)    
    # plt.plot(res_traff_mon, label='Reserved traff', linewidth=.8)    
    # plt.plot(realTraf_sum, label='Real traff', linewidth=.8)    
    # plt.grid(), plt.legend(), plt.show()  

    # n_samples_M = 240
    # n_samples_m = 0
    # plt.plot(reqTraf_sum[n_samples_m:n_samples_M], label='Requested traff', linewidth=.5)    
    # plt.plot(res_traff_mon[n_samples_m:n_samples_M], label='Reserved traff', linewidth=.5)    
    # plt.plot(realTraf_sum[n_samples_m:n_samples_M], label='Real traff', linewidth=.5)    
    # plt.grid(), plt.legend(), plt.show()

    # #%% plt.plot()
    # for service in range(5):
    #     xx = x_sol[:,service]    
    #     zz = z_sol[:,service]

    #     resTraff_ser = np.kron(xx*zz, np.ones(T_mno))
    #     realTraf_ser = realTraf.iloc[:,service]
    #     reqTraf_ser  = reqTraf.iloc[:,service]

    #     # plt.plot(reqTraf_ser, label='Requested traff')    
    #     # plt.plot(resTraff_ser, label='Reserved traff')    
    #     # plt.plot(realTraf_ser, label='Real traff')    
    #     # plt.grid(), plt.legend(), plt.show()  

    #     plt.plot(reqTraf_ser[:1000], label='Requested traff')    
    #     plt.plot(resTraff_ser[:4*1000:4], label='Reserved traff')    
    #     plt.plot(realTraf_ser[:1000], label='Real traff')    
    #     plt.grid(), plt.legend(), plt.show()
            

    # #     plt.plot()
Ctots=[1]

pair_list_duration_OP=[]
synthetic_dataset = 0
T_mnos = [5,15,30,60,120]
doFiles = 0

for T_mno in T_mnos:
    for Ctot in Ctots:
        pair_list_duration_OP.append([synthetic_dataset,Ctot,T_mno,doFiles])

    


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(myMultiOpt,pair_list_duration_OP)