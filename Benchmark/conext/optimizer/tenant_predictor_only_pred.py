# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:49:58 2022

@author: antonio
"""

import sys
sys.path.append('../helpers/')

import os
import numpy as np
import math
from helpers import *
import time


def do(mypredictor, topo_file, tenants_file, Tdec, monitoring_samples):
    print("############# Tenant Traffic Predictor ####################")
    if mypredictor == "lstm":
        import tenant_predictor_lstm as predictor
    elif mypredictor == "holtwinters":
        import tenant_predictor_holtwinters as predictor
    else:
        raise Exception("ERRROR: Incorrect predictor name")
    
    ######## 0.1) Topology
    myinfrastructure = Infrastructure(Network(nx.Graph()) , [], [])
    myinfrastructure.fromJSON(topo_file)  
    NumBS = len(myinfrastructure.base_stations)

    #0.2) Tenant requests
    alltenants = AllTenants()
    alltenants.fromJSON(tenants_file)
    mytenants = alltenants.tenant
    NumTenants = len(mytenants)
    forecast_list=np.zeros(NumTenants)
    forecast_uncertainty=0
    #Generate FORECAST
    for tau in range(0,NumTenants):
        print("Tenant {0}: ".format(mytenants[tau].idx))        
        idx_BS = myinfrastructure.base_stations[0].idx 
        
        
        # for i in range( 0 ,NumBS):
        TrainingTraffic = [mytenants[tau].data_history[str(idx_BS)][str(k)] for k in range(len(mytenants[tau].data_history[str(idx_BS)]))] # get history data in order
        
        myrequest = float(mytenants[tau].request.bitrate[str(idx_BS)])
        
        forecast, forecast_uncertainty = predictor.make_forecast(TrainingTraffic, myrequest, mytenants[tau].idx,  Tdec) #   str(idx_BS),
        
        ### no forecasting
        forecast = TrainingTraffic[-1]
        forecast_uncertainty = 0 ##################################################################################################################
        
        mytenants[tau].forecast_usage[str(idx_BS)] = min(forecast, float(mytenants[tau].request.bitrate[str(idx_BS)]))
        mytenants[tau].forecast_uncertainty[str(idx_BS)] = min(forecast_uncertainty,  float(mytenants[tau].request.bitrate[str(idx_BS)]))            

        print("BS {0}, forecast = {1} + {2} Mb/s".format(idx_BS, mytenants[tau].forecast_usage[str(idx_BS)], mytenants[tau].forecast_uncertainty[str(idx_BS)])) 
        forecast_list[tau]=mytenants[tau].forecast_usage[str(idx_BS)]  
    # Write tenants file        
    for tau in range(0, NumTenants):
        for i in range( 0, NumBS):
            if len(mytenants[tau].data_history[str(idx_BS)]) < monitoring_samples: # clean it...
                print('\nCLEANING HISTORY PLEASE CHECK!!\n')            
                mytenants[tau].data_history[str(idx_BS)] = []
                    
    alltenants.toJSON(tenants_file)
    
    return forecast_list, forecast_uncertainty


#%%
def dos(mypredictor, topo_file, tenants_file, Tdec, monitoring_samples):
    print("############# Tenant Traffic Predictor ####################")
    if mypredictor == "lstm":
        import tenant_predictor_lstm as predictor
    elif mypredictor == "holtwinters":
        import tenant_predictor_holtwinters as predictor
    else:
        raise Exception("ERRROR: Incorrect predictor name")
    
    ######## 0.1) Topology
    myinfrastructure = Infrastructure(Network(nx.Graph()) , [], [])
    myinfrastructure.fromJSON(topo_file)  
    NumBS = len(myinfrastructure.base_stations)

    #0.2) Tenant requests
    alltenants = AllTenants()
    alltenants.fromJSON(tenants_file)
    mytenants = alltenants.tenant
    NumTenants = len(mytenants)

    #Generate FORECAST
    for tau in range(0,NumTenants):
        print("Tenant {0}: ".format(mytenants[tau].idx))        
        idx_BS = myinfrastructure.base_stations[0].idx 
        
        # for i in range( 0 ,NumBS):
        TrainingTraffic = [mytenants[tau].data_history[str(idx_BS)][str(k)] for k in range(len(mytenants[tau].data_history[str(idx_BS)]))] # get history data in order
        
        myrequest = float(mytenants[tau].request.bitrate[str(idx_BS)])
        
        forecast, forecast_uncertainty = predictor.make_forecast(TrainingTraffic, myrequest, mytenants[tau].idx,  Tdec) #   str(idx_BS),
        
        ### no forecasting
        # forecast = TrainingTraffic[-1]
        # forecast_uncertainty = 0 ##################################################################################################################
        
        mytenants[tau].forecast_usage[str(idx_BS)] = min(forecast, float(mytenants[tau].request.bitrate[str(idx_BS)]))
        mytenants[tau].forecast_uncertainty[str(idx_BS)] = min(forecast_uncertainty,  float(mytenants[tau].request.bitrate[str(idx_BS)]))            

        print("BS {0}, forecast = {1} + {2} Mb/s".format(idx_BS, mytenants[tau].forecast_usage[str(idx_BS)], mytenants[tau].forecast_uncertainty[str(idx_BS)])) 
            
    # Write tenants file        
    for tau in range(0, NumTenants):
        for i in range( 0, NumBS):
            if len(mytenants[tau].data_history[str(idx_BS)]) < monitoring_samples: # clean it...            
                mytenants[tau].data_history[str(idx_BS)] = []
                    
    alltenants.toJSON(tenants_file)


   

