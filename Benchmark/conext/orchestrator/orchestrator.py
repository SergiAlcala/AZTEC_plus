#!/usr/bin/python
import sys
sys.path.append('../helpers/')
from helpers import *
import os
import numpy as np
import math
import time

# import mysql.connector
# import wget
from shutil import copyfile 

def update_tenant_samples(topo_file, tenants_file, ResultFile, monitoring_samples):

    it_counter = 0
    agg_reward = 0
    agg_penalty = 0

    #0.1) Topology
    myinfrastructure = Infrastructure(Network(nx.Graph()) , [], [])
    myinfrastructure.fromJSON(topo_file)  
    NumBS = len(myinfrastructure.base_stations)

    #0.2) Tenant requests
    alltenants = AllTenants()
    alltenants.fromJSON(tenants_file)
    mytenants = alltenants.tenant

    NumTenants = len(mytenants)

    print("############# Interval = {} Updating... ####################".format(alltenants.decision_interval))
    #print("Time = {}".format(t))
    for tau in range(0, NumTenants):
        if mytenants[tau].accepted == 0:  ## TBD: UNCOMENT FOR REAL SCENARIO!!!!!!!!!!!!!
            continue 
        #print("Tenant {0}: ".format(mytenants[tau].idx))
        for i in range( 0, NumBS):
            idxBS    = str(myinfrastructure.base_stations[i].idx)
            
            Lambda   = mytenants[tau].request.bitrate[idxBS]
            lambda_i = mytenants[tau].service.mean_bitrate[idxBS]
            sigma_i  = mytenants[tau].service.std_bitrate[idxBS]
            
            if len(mytenants[tau].data_history[idxBS]) > 0:
                Traffic = [ mytenants[tau].data_history[idxBS][str(k)] for k in range(len(mytenants[tau].data_history[idxBS])) ] # get history data in order            
            else:
                Traffic = []
                
            # for t in range(mon_samples): # simulate traffic

            ### updating history with new samples. ###############
            ## next traffic 
            Traffic_future = [mytenants[tau].data_future[idxBS][str(k)] for k in range(len(mytenants[tau].data_future[idxBS]))]
    
            updatedTraffic = Traffic[monitoring_samples:] +  Traffic_future[:monitoring_samples]
            Traffic_future = Traffic_future[monitoring_samples:]
            
            mytenants[tau].data_history[idxBS] = dict(zip([str(e) for e in range(len(updatedTraffic))], updatedTraffic))
            mytenants[tau].data_future[idxBS] = dict(zip([str(e) for e in range(len(Traffic_future))], Traffic_future))
            
            #print("BS {0}, new sample = {1} Mb/s".format(myinfrastructure.base_stations[i].idx, sample)) 

    saveAndPrint = False
    if saveAndPrint:            
        print("############# Interval = {} Saving... ####################".format(alltenants.decision_interval))
    
        #plotting
        #print("time\ttau\tduration\tbs_id\tthr\tmux.g\tpenalty")
        agg_ran = 0
        for t in range(len(Traffic)): # simulate traffic
            it_counter = it_counter + 1
            count_accepted = 0
            for tau in range(0, NumTenants):
                if mytenants[tau].accepted == 0:
                    continue
                count_accepted = count_accepted +1
                for i in range( 0, NumBS):
                    agg_ran = agg_ran + mytenants[tau].nw_alloc[idxBS]
                    
                    Traffic = [ mytenants[tau].data_history[idxBS][str(k)] for k in range(len(mytenants[tau].data_history[idxBS])) ] # get history data in order
                    current_traffic = Traffic[-len(Traffic)+t]
                    agg_request = 0
                    agg_reserved = 0
                    
                    for tau2 in range(0, NumTenants):
                        if mytenants[tau2].accepted == 0:
                            continue
                        agg_capacity = 0
                        for i2 in range( 0, NumBS):
                            agg_request = agg_request + float(mytenants[tau2].request.bitrate[str(myinfrastructure.base_stations[i2].idx)])
                            agg_reserved = agg_reserved + mytenants[tau2].nw_alloc[str(myinfrastructure.base_stations[i2].idx)]
                            agg_capacity = agg_capacity + float(myinfrastructure.base_stations[i2].capacity)
                    mux_gain = agg_request/float(min(agg_capacity, agg_reserved))
                    penalty = max(0, float(current_traffic) - float(mytenants[tau].nw_alloc[idxBS]))
                    
                    #print("{0}\t{1}\t{2}\t{3}\t{4}\t{5:0.01f}\t{6}\t{7}\t{8}\t{9}\t{10}".format(t, mytenants[tau].idx, mytenants[tau].request.duration, myinfrastructure.base_stations[i].idx, float(current_traffic), mux_gain, penalty, float(mytenants[tau].request.bitrate[idxBS]), mytenants[tau].nw_alloc[idxBS], agg_request, agg_reserved))    
                    
                    #TBD: ESTO ES PARA EL SERVIDOR QUE PLOTEE: HTTP POST AL CONTROLADOR PRINCIPAL
                    
                    agg_penalty = agg_penalty + float(mytenants[tau].penalty)*penalty
                    agg_reward = agg_reward + float(mytenants[tau].reward)
    
                    f = open(ResultFile, 'a')
                    f.write(str(alltenants.decision_interval) + "\t" + str(t) + "\t" + str(mytenants[tau].idx) + "\t" + str(i) + "\t" + str(mytenants[tau].dc_alloc) + "\t" + str(mytenants[tau].nw_alloc[idxBS]) +   "\t"  +  str(mytenants[tau].service.alpha) +   "\t"   +  str(mytenants[tau].service.beta) + "\t"  + str(mytenants[tau].request.bitrate[idxBS]) + "\t" +  str(mytenants[tau].reward) + "\t" +  str(mytenants[tau].penalty) + "\t" + str(mytenants[tau].service.std_bitrate[idxBS]) + "\t" + str(count_accepted) + "\t" + str(agg_reward) + "\t" + str(agg_penalty) + "\t" +  str(mux_gain) + "\t" + str(current_traffic) + "\n")
                    f.close()
            #time.sleep(1)
            #print('----------------------------------------------------')
        #print('SUMMARY')
        if(count_accepted<1):
            mux_gain = 0
            penalty = 0
            
        print('iteration={0}\tmean reward={1:0.1f}\tmean penalty={2:0.1f}\tmean net gain={3:0.1f}\tagg RAN capacity={4:.01f}'.format(it_counter, agg_reward/it_counter, agg_penalty/it_counter, (agg_reward-agg_penalty)/it_counter, agg_ran/it_counter))
        print('----------------------------------------------------')
        #f = open(ResultFile, 'a')
        #f.write(str(count_accepted) + "\t" + str(agg_reward/it_counter) + "\t" + str(agg_penalty/it_counter) + "\t" +  str(mux_gain) + "\n")
        #f.close()
        #raw_input("Press key to continue...")
        
        
    ###########################################################################
    
    #change state...
    for tau in range(len(mytenants)):
        if mytenants[tau].status == "ACCEPTED":
            if int(mytenants[tau].request.duration_remaining) > 0:
                mytenants[tau].request.duration_remaining = int(mytenants[tau].request.duration_remaining) - 1
            if  mytenants[tau].request.duration_remaining == 0: 
                print(f'Tenant {tau} is finished. Remaining duration = 0')
                if int(mytenants[tau].request.loop) == 1:
                    mytenants[tau].request.duration_remaining = mytenants[tau].request.duration
                    print("changing to pending")
                    mytenants[tau].status = "PENDING"
                    mytenants[tau].accepted = 0
                else:                            
                    mytenants[tau].status = "OFFLINE"
                    mytenants[tau].accepted = 0
                #mytenants[tau].request.duration = 1    
            else:
                print(f'Tenant {tau} is not finished. Remaining duration = {mytenants[tau].request.duration_remaining}')
        
    # write tenant file                
    alltenants.toJSON(tenants_file)

import requests

def sync_dashboard(tenants_file, tenant_dashboard_file): 
    print("############# SYNCHRONIZING DASHBOARD DATA... ####################")
    copyfile(tenants_file, tenant_dashboard_file)

    #alltenants = AllTenants()
    #alltenants.fromJSON(tenants_file) # get from dashboard
    #mytenants = alltenants.tenant
 
    ## api-endpoints
    #URLupdatedatabase = "http://localhost:8080/REST-API-LinuxVM/webapi/updatedatabase"
    #URLdelete = "http://localhost:8080/REST-API-LinuxVM/webapi/deleterejected"
    #URLcreatejson = "http://localhost:8080/REST-API-LinuxVM/webapi/createjson"
    #URLgetjson = "http://localhost:8080/REST-API-LinuxVM/webapi/getjson"
   
    #headers = {'content-type': 'text/plain'} #same header for all the http post
      
    ## sending get request and saving the response as response object
   
    #for tau in range(0, len(mytenants)):
        #print("Updating tenant {}".format(mytenants[tau].idx))
   
        #if mytenants[tau].status == "REJECTED":
            #print("CHANGING TO REJECT!!!!")
            #str_ = str(mytenants[tau].idx) +"&3"
            #r = requests.post(url = URLupdatedatabase, data = str_, headers=headers)
            ##pastebin_url = r.text
            ##print("The pastebin URL is:%s"%pastebin_url)
        #elif mytenants[tau].status == "ACCEPTED": 
            #print("CHANGING TO ACCEPT!!!!")
            #str_ = str(mytenants[tau].idx) +"&1"
            #r = requests.post(url = URLupdatedatabase, data = str_, headers=headers)
        #elif  mytenants[tau].status == "OFFLINE": # Delete offline tenants from mysql database
            #print("CHANGING TO OFFLINE!!!!")
            #str_ = str(mytenants[tau].idx) +"&4"
            #r = requests.post(url = URLupdatedatabase, data = str_, headers=headers)   
        #elif mytenants[tau].status == "PENDING":  # should only happen for loop=1
            #str_ = str(mytenants[tau].idx) +"&2"
            #r = requests.post(url = URLupdatedatabase, data = str_, headers=headers)
        #else:
            #print("STATUS IS {}".format(mytenants[tau].status))
                   

 
    ##SIGNAL DASHBOARD TO READ DATABASE AND PRODUCE A NEW FILE
    #g = requests.get(url = URLdelete, headers=headers) # cleanup database
    #g = requests.get(url = URLcreatejson, headers=headers) # Force creation of new JSON file

def sync_orchestrator(topo_file, tenant_dashboard_file, tenants_file, arrivalTime):

    print("\n############# syncrhonizing data with dashboard... ##############")

    MEMORY_SIZE = 1000000 # how many decision intervals I keep data history of a tenant after being offline

    ## 1) Topology
    myinfrastructure = Infrastructure()
    myinfrastructure.fromJSON(topo_file)
    NumBS = len(myinfrastructure.base_stations)

    allnewtenants = AllTenants()
    if arrivalTime:
        allnewtenants.fromJSON(tenant_dashboard_file) # get from dashboard
    mynewtenants = allnewtenants.tenant

    allcurrenttenants = AllTenants()
    mycurrenttenants = allcurrenttenants.tenant

    # is there new tenants? 
    if not os.path.exists(tenants_file): # all are new :)
        print(' +-+-+-+  All Tenants are new +-+-+-+')
        # print(len(mynewtenants))
        for tau in range(0, len(mynewtenants)): # initialize new tenants
            print("Adding new tenant ID {}".format(mynewtenants[tau].idx))
            mycurrenttenants.append(mynewtenants[tau])
            for i in range( 0, NumBS):
                # mycurrenttenants[-1].data_history[str(myinfrastructure.base_stations[i].idx)] = []
                mycurrenttenants[-1].nw_alloc[str(myinfrastructure.base_stations[i].idx)] = []  
                
                        
    else: # add the new tenants. Remove old
        
        allcurrenttenants = AllTenants()
        allcurrenttenants.fromJSON(tenants_file) 
        mycurrenttenants = allcurrenttenants.tenant
            
        new_idx_list = []
        for tau in range(0, len(mynewtenants)):
            new_idx_list.append(mynewtenants[tau].idx)
            
        # remove non-existent tenants
        allnewcurrenttenants = AllTenants()
        mynewcurrenttenants = allnewcurrenttenants.tenant    
        
        toremove = []
        for tau in range(0, len(mycurrenttenants)):
            if mycurrenttenants[tau].status != "ACCEPTED":
                if not mycurrenttenants[tau].idx in new_idx_list:
                    print("Removing tenant ID {}".format(mycurrenttenants[tau].idx))
                    toremove.append(tau)

            if int(allcurrenttenants.decision_interval) - int(mycurrenttenants[tau].last_accepted)  > MEMORY_SIZE:
                print("Removing OLD tenant ID {} (last_accepted={}, current_interval={})".format(mycurrenttenants[tau].idx, mycurrenttenants[tau].last_accepted, allcurrenttenants.decision_interval))
                toremove.append(tau) 
        
        for tau in range(0, len(mycurrenttenants)):    
            if not tau in toremove:
                mynewcurrenttenants.append(mycurrenttenants[tau])
            #else:
                #print("Removing tenant ID {}".format(mycurrenttenants[tau].idx))
                
        allnewcurrenttenants.decision_interval = allcurrenttenants.decision_interval
        allcurrenttenants = allnewcurrenttenants
        mycurrenttenants = allcurrenttenants.tenant    
        
        current_idx_list = []
        for tau in range(0, len(mycurrenttenants)):
            current_idx_list.append(mycurrenttenants[tau].idx)    
                
        
        # update existing tenants
        for tau in range(0, len(mynewtenants)):
            if mynewtenants[tau].idx in current_idx_list:
                #update
                curr_tau = current_idx_list.index(mynewtenants[tau].idx)
                print("Updating info of tenant ID {} state={}".format(mynewtenants[tau].idx, mycurrenttenants[curr_tau].status))
                
                mycurrenttenants[curr_tau].service = mynewtenants[tau].service
                
                if mycurrenttenants[curr_tau].status != "ACCEPTED":
                    mycurrenttenants[curr_tau].penalty = mynewtenants[tau].penalty
                    mycurrenttenants[curr_tau].reward = mynewtenants[tau].reward
                    mycurrenttenants[curr_tau].request = mynewtenants[tau].request
                    mycurrenttenants[curr_tau].request.duration_remaining = mycurrenttenants[curr_tau].request.duration
                    mycurrenttenants[curr_tau].status = "PENDING"


                
        # add new tenants
        for tau in range(0, len(mynewtenants)):    
            if not mynewtenants[tau].idx in current_idx_list:     
                if mynewtenants[tau].status == "PENDING":
                    print("Adding new tenant ID {}".format(mynewtenants[tau].idx))
                    mycurrenttenants.append(mynewtenants[tau])
                    mycurrenttenants[tau].request.remaining_duration = mycurrenttenants[tau].request.duration                    
                    for i in range( 0, NumBS):
                        # mycurrenttenants[-1].data_history[str(myinfrastructure.base_stations[i].idx)] = []
                        mycurrenttenants[-1].nw_alloc[str(myinfrastructure.base_stations[i].idx)] = []      
                        

        

    for tau in range(0, len(mycurrenttenants)):
        print("Tenant ID {} STATE = {} (duration={})".format(mycurrenttenants[tau].idx, mycurrenttenants[tau].status, mycurrenttenants[tau].request.duration))
                
    allcurrenttenants.toJSON(tenants_file)

    
def sync_orchestrator_old(topo_file, tenant_dashboard_file, tenants_file):

    print("\n############# syncrhonizing data with dashboard... ##############")

    MEMORY_SIZE = 1000000 # how many decision intervals I keep data history of a tenant after being offline

    ## 1) Topology
    myinfrastructure = Infrastructure()
    myinfrastructure.fromJSON(topo_file)
    NumBS = len(myinfrastructure.base_stations)

    allnewtenants = AllTenants()
    allnewtenants.fromJSON(tenant_dashboard_file) # get from dashboard
    mynewtenants = allnewtenants.tenant

    allcurrenttenants = AllTenants()
    mycurrenttenants = allcurrenttenants.tenant

    # is there new tenants? 
    if not os.path.exists(tenants_file): # all are new :)
        print(' +-+-+-+  All Tenants are new +-+-+-+')
        # print(len(mynewtenants))
        for tau in range(0, len(mynewtenants)): # initialize new tenants
            print("Adding new tenant ID {}".format(mynewtenants[tau].idx))
            mycurrenttenants.append(mynewtenants[tau])
            for i in range( 0, NumBS):
                # mycurrenttenants[-1].data_history[str(myinfrastructure.base_stations[i].idx)] = []
                mycurrenttenants[-1].nw_alloc[str(myinfrastructure.base_stations[i].idx)] = []  
                
                        
    else: # add the new tenants. Remove old
        
        allcurrenttenants = AllTenants()
        allcurrenttenants.fromJSON(tenants_file) 
        mycurrenttenants = allcurrenttenants.tenant
            
        new_idx_list = []
        for tau in range(0, len(mynewtenants)):
            new_idx_list.append(mynewtenants[tau].idx)
            
        # remove non-existent tenants
        allnewcurrenttenants = AllTenants()
        mynewcurrenttenants = allnewcurrenttenants.tenant    
        
        toremove = []
        for tau in range(0, len(mycurrenttenants)):
            if mycurrenttenants[tau].status != "ACCEPTED":
                if not mycurrenttenants[tau].idx in new_idx_list:
                    print("Removing tenant ID {}".format(mycurrenttenants[tau].idx))
                    toremove.append(tau)

            if int(allcurrenttenants.decision_interval) - int(mycurrenttenants[tau].last_accepted)  > MEMORY_SIZE:
                print("Removing OLD tenant ID {} (last_accepted={}, current_interval={})".format(mycurrenttenants[tau].idx, mycurrenttenants[tau].last_accepted, allcurrenttenants.decision_interval))
                toremove.append(tau) 
        
        for tau in range(0, len(mycurrenttenants)):    
            if not tau in toremove:
                mynewcurrenttenants.append(mycurrenttenants[tau])
            #else:
                #print("Removing tenant ID {}".format(mycurrenttenants[tau].idx))
                
        allnewcurrenttenants.decision_interval = allcurrenttenants.decision_interval
        allcurrenttenants = allnewcurrenttenants
        mycurrenttenants = allcurrenttenants.tenant    
        
        current_idx_list = []
        for tau in range(0, len(mycurrenttenants)):
            current_idx_list.append(mycurrenttenants[tau].idx)    
                
        
        # update existing tenants
        for tau in range(0, len(mynewtenants)):
            if mynewtenants[tau].idx in current_idx_list:
                #update
                curr_tau = current_idx_list.index(mynewtenants[tau].idx)
                print("Updating info of tenant ID {} state={}".format(mynewtenants[tau].idx, mycurrenttenants[curr_tau].status))
                
                mycurrenttenants[curr_tau].service = mynewtenants[tau].service
                
                if mycurrenttenants[curr_tau].status != "ACCEPTED":
                    mycurrenttenants[curr_tau].penalty = mynewtenants[tau].penalty
                    mycurrenttenants[curr_tau].reward = mynewtenants[tau].reward
                    mycurrenttenants[curr_tau].request = mynewtenants[tau].request
                    mycurrenttenants[curr_tau].request.duration_remaining = mycurrenttenants[curr_tau].request.duration
                    mycurrenttenants[curr_tau].status = "PENDING"


                
        # add new tenants
        for tau in range(0, len(mynewtenants)):    
            if not mynewtenants[tau].idx in current_idx_list:     
                if mynewtenants[tau].status == "PENDING":
                    print("Adding new tenant ID {}".format(mynewtenants[tau].idx))
                    mycurrenttenants.append(mynewtenants[tau])
                    mycurrenttenants[tau].request.remaining_duration = mycurrenttenants[tau].request.duration                    
                    for i in range( 0, NumBS):
                        # mycurrenttenants[-1].data_history[str(myinfrastructure.base_stations[i].idx)] = []
                        mycurrenttenants[-1].nw_alloc[str(myinfrastructure.base_stations[i].idx)] = []      
                        

        

    for tau in range(0, len(mycurrenttenants)):
        print("Tenant ID {} STATE = {} (duration={})".format(mycurrenttenants[tau].idx, mycurrenttenants[tau].status, mycurrenttenants[tau].request.duration))
                
    allcurrenttenants.toJSON(tenants_file)

def sync_nw_controller():
	print("Starting up network slicing...")
	#TBD: call script  to configure generators through ssh...


