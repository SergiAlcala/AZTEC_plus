#!/usr/bin/python
import sys
sys.path.append('../helpers/')

import os
import numpy as np
import math
from helpers import *
import time

#predictor = "lstm" # lstm/holtwinters


def do(mypredictor):

    print("############# Tenant Traffic Predictor ####################")
    if mypredictor == "lstm":
        import tenant_predictor_lstm as predictor
    elif mypredictor == "holtwinters":
        import tenant_predictor_holtwinters as predictor
    else:
        raise Exception("ERRROR: Incorrect predictor name")
        
    
    ######## 0.1) Topology
    G         = nx.Graph()
    mynetwork = Network(G) 
    base_station = []
    dc = []
    myinfrastructure = Infrastructure(mynetwork, base_station, dc)
    myinfrastructure.fromJSON(topo_file)
    NumBS = len(myinfrastructure.base_stations)

    #0.2) Tenant requests
    alltenants = AllTenants()

    alltenants = AllTenants()
    alltenants.fromJSON(tenants_file)
    mytenants = alltenants.tenant
    NumTenants = len(mytenants)


    #Generate  FORECAST
    for tau in range(0,NumTenants):
        print("Tenant {0}: ".format(mytenants[tau].idx))
        for i in range( 0 ,NumBS):
            #if mytenants[tau].data_history == ""
            TrainingTraffic = [ mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)][str(k)] for k in range(len(mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)])) ] # get history data in order
            myrequest = float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])
            
            forecast, forecast_uncertainty = predictor.make_forecast(TrainingTraffic, myrequest, mytenants[tau].idx, str(myinfrastructure.base_stations[i].idx) );   
            
            # no forecasting
            ##forecast = mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]
            #forecast_uncertainty = 0
            
            mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)] = min(forecast, float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]))
            mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[i].idx)] = min(forecast_uncertainty,  float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]))            

            print("BS {0}, forecast = {1} + {2} Mb/s".format(myinfrastructure.base_stations[i].idx, mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)], mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[i].idx)])) 
            
    # Write tenants file        
    for tau in range(0, NumTenants):
        for i in range( 0, NumBS):
            if len(mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)]) < mon_samples: # clean it...            
                mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)] = []
                    
    alltenants.toJSON(tenants_file)


