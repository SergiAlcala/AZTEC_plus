from __future__ import print_function

import sys, os

if os.path.join('..','helpers') not in sys.path:
    sys.path.append(os.path.join('..','helpers')) # os.getcwd()

from subprocess import call
import re
import numpy as np
import time

import solver
import benders
import networkx as nx
from helpers import *


def do(topo_file, tenants_file, leasing_cost):
    print("############# Compute Optimal Allocation ####################")
    
    ## 0) Read data from json file (TBD: Received from socket?)
    #0.1) Topology
    myinfrastructure = Infrastructure(Network(nx.Graph()) , [], [])
    myinfrastructure.fromJSON(topo_file)  
    
    myinfrastructure.leasing_cost = leasing_cost 
    
    #0.2) Tenant requests
    alltenants = AllTenants()
    alltenants.fromJSON(tenants_file)
    mytenants = alltenants.tenant
    
    # 2) remove offline tenants 
    #TBD: CHECK REQUEST CONSISTENCY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    my_online_tenants = []
    list_online_tenants_idx = []
    for tau in range(len(mytenants)):
        if mytenants[tau].status != "OFFLINE":
            my_online_tenants.append(mytenants[tau])
            list_online_tenants_idx.append(mytenants[tau].idx)
            
    print("{} ONLINE tenants".format(len(my_online_tenants)))
    print(list_online_tenants_idx)
	

    if len(my_online_tenants) > 0:
        ## 3) Solve it ##
        zsol, xsol, usol, vsol, wsol, myindex = solver.solve(my_online_tenants, myinfrastructure)
        #zsol, xsol, usol, vsol, wsol, myindex = benders.solve(my_online_tenants, myinfrastructure)
    
        print("############# SOLUTION ####################")

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
        for link_ in myinfrastructure.network.graph.edges(data=True):
            if wsol[i] > 1e-2:
                print("link {0} leased capacity={1:0.01f} Mb/s".format(link_, wsol[i]))
                #variable = raw_input('Please sth to continue')
            linkid = linkid + 1
        

    else:
        print("There's nothing to do...")
        
    #update interval counter and write to json file    
    alltenants.decision_interval = alltenants.decision_interval + 1
    alltenants.toJSON(tenants_file)
    
    return zsol, xsol