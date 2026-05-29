from __future__ import print_function

import sys
sys.path.append('/home/jupyter-sergi/2023_02_03 - CoNEXT benchmark - shared/helpers/')
import imp
import cplex
from cplex.callbacks import UserCutCallback, LazyConstraintCallback
from cplex.exceptions import CplexError
from numpy import *
import time
import os
#from scipy.stats import genpareto
#from scipy.stats import pareto
#import matplotlib.pyplot as plt

#from decorator import append
import scipy.io as io
import networkx as nx
from helpers import *

        

def createMILP(cpx, mytenants, myinfrastructure, z, x, y, u, v, w):
    #### Create variables and objective linear coefficients
    cpx.objective.set_sense(cpx.objective.sense.minimize)    
    
    myindex = {}
    longest_path = 0
    shortest_path = 1e9
    # z: bitrate 
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):
                    
                    # figure out longest and shortest path to fine tune rewards later on
                    if len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p])>longest_path:
                        longest_path = len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p])
                        
                    if len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p])<shortest_path:
                        shortest_path = len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p])
                        
                    # add variable and cost factor associated in the obj function
                    varName = "z" + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    z.append(cpx.variables.get_num())
                    myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)] = cpx.variables.get_num() # map to access variables later 
                    
                    cpx.variables.add(obj = [0.0], 
                                lb=[0.0], ub=[float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])], types=["C"],
                                names=[varName]) 
    #print("longest path={}, shortest_path={}".format(longest_path, shortest_path))
    
    # x: access control and routing          
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):                    
                    
                    max_path_penalty = 0.001 # relative reward (between 0 and 1) reduction for longest penalty <- This should be small but larger than 0
                    min_path_penalty = 0 # relative reward (between 0 and 1) reduction for shortest penalty (reward reduction on other path lengths are proportional)
                    
                    b = (max_path_penalty - min_path_penalty)#/(longest_path - shortest_path) # slope
                    a = min_path_penalty -b*shortest_path # offset
                    
                    mypath_penalty = a + b*len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p]) # reward is penalized depending on the path length
                    
                    
                    uncertainty = mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[i].idx)]/float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])
                    
                    #print("uncertainty = {}".format(uncertainty))
                    #duration = mytenants[tau].request.duration
                    duration = 4
                    rho_hat = (mytenants[tau].penalty*uncertainty*duration)/(float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]) - mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)]+ sys.float_info.epsilon)   
#                    print("tau={},bs={} risk = {}, request={}, forecast={}, norm_factor={}".format(mytenants[tau].idx, i, risk1*risk2, mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)], mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)], norm_factor))

                    if uncertainty == 0:
                        rho_hat = 0
                    
                    #print("rho_hat = {}".format(rho_hat))
                    varName = "x" + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    x.append(cpx.variables.get_num())
                    cpx.variables.add(obj = [ rho_hat \
                                              * float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]) \
                                              - float(mytenants[tau].reward) ], 
                                lb=[0], ub=[1], types=["B"],
                                names=[varName])      
                    
    # y: auxiliary variable to linearize quadratic objective
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):                    
                    
                    varName = "y" + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    y.append(cpx.variables.get_num())
                    
                    uncertainty = mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[i].idx)]/float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])
                    #duration = mytenants[tau].request.duration
                    duration = 1
                    rho_hat = (mytenants[tau].penalty*uncertainty*duration)/(float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]) - mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)]+ sys.float_info.epsilon)   
                    
                    if uncertainty == 0:
                        rho_hat = 0
                                        
                    #print("rho_hat = {}".format(rho_hat))
                    cpx.variables.add(obj = [ -1.0*rho_hat ], 
                                lb=[0], ub=[float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])], types=["C"],
                                names=[varName])          
                    
                    
    # u: additional leasing on CU capacity 
    leasing_cost = 100
    for d in range(len(myinfrastructure.dcs)):
                
        varName = "u" + str(d) 
        u.append(cpx.variables.get_num())
        
        cpx.variables.add(obj = [ 1.0*leasing_cost ], 
                    lb=[0], ub=[1e6], types=["C"],
                    names=[varName])                       
                    
    # v: additional leasing on BS capacity 
    leasing_cost = 100
    for i in range(len(myinfrastructure.base_stations)):
                
        varName = "v" + str(i) 
        v.append(cpx.variables.get_num())
        
        cpx.variables.add(obj = [ 1.0*leasing_cost ], 
                    lb=[0], ub=[150], types=["C"],
                    names=[varName])          
                
    # w: additional leasing on network link capacity 
    leasing_cost = 100
    nlinks = myinfrastructure.network.graph.number_of_edges()
    linkid=0
    for link_ in myinfrastructure.network.graph.edges(data=True):
        
        varName = "w" + str(linkid) 
        w.append(cpx.variables.get_num())
        
        cpx.variables.add(obj = [ 1.0*leasing_cost ], 
                    lb=[0], ub=[1e6], types=["C"],
                    names=[varName])          
                                    
    linkid = linkid + 1
                   
    #print(cpx.objective.get_linear())
    #print(cpx.variables.get_names())

    #### Create constraints
    
    # 1. Capacity constraints on CUs
    
    #beta = 1.0
    #alpha = 0.0

    alpha_all = 0.0
    for d in range(len(myinfrastructure.dcs)):
        thevars = []
        thecoefs = []
        for tau in range(len(mytenants)):
            alpha = mytenants[tau].service.alpha
            beta = mytenants[tau].service.beta
            
            print("alpha={}, beta={}, capacity = {}".format(alpha, beta, myinfrastructure.dcs[d].capacity))
            if (alpha == 0) and (beta == 0):
                continue
            
            alpha_all = alpha_all + alpha
            for i in range(len(myinfrastructure.base_stations)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):    
                    thevars.append(z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0*float(beta))
                    

        thevars.append(u[d]) # additional capacity needed
        thecoefs.append(-1.0)
        cName = "c1." + str(d) 


        #print(cpx.linear_constraints.get_rows())
        #print(thecoefs)
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[myinfrastructure.dcs[d].capacity - alpha_all], names=[cName])    
        
    ## 2. Capacity constraints on network links

    eta = 1.0

    nlinks = myinfrastructure.network.graph.number_of_edges()
    linkid=0
    for link_ in myinfrastructure.network.graph.edges(data=True):
        thevars = []
        thecoefs = []        
        link = (link_[0], link_[1])
        link_att = link_[2]
        
        #print("link {0} has capacity {1}".format(link,link_att['cap']))
        unused_link = 1
        for tau in range(len(mytenants)):
            for i in range(len(myinfrastructure.base_stations)):
                for d in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):  
                        if IsLinkInPath(link, myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p]):
                            #print(tau)
                            #print(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][p])
                            if (z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]] in thevars): # TBD we must ensure that eta is the same as well!!!
                                continue
                                
                            thevars.append(z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                            thecoefs.append(1.0*eta)
                            unused_link = 0
                            
        if unused_link == 1:
            continue
        
        thevars.append(w[linkid]) # additional capacity needed
        linkid = linkid + 1        
        thecoefs.append(-1.0)                            
                            
        cName = "c2." + str(linkid) 
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[link_att['cap']], names=[cName])     
        
    ## 3. Capacity constraints on BSs
    eta = 1.0
    
    for i in range(len(myinfrastructure.base_stations)):
        
        thevars = []
        thecoefs = []
        for tau in range(len(mytenants)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):    
                    thevars.append(z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0*eta)
                    

        thevars.append(v[i]) # additional capacity needed
        thecoefs.append(-1.0)
        
        cName = "c3." + str(i) 

        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[myinfrastructure.base_stations[i].capacity], names=[cName])         
    
    ## 4. just one path per tenant and BS, and one CU
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
        
            thevars = []
            thecoefs = []         
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):    
                    thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0)
                    
            cName = "c4." + str(tau) + "." + str(i)  
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                senses=["L"], rhs=[1.0], names=[cName])            
            
    ## 5. if accepted, do so in all BSs and use the same DC
    for tau in range(len(mytenants)):
        for d in range(len(myinfrastructure.dcs)):
            for i in range(len(myinfrastructure.base_stations)):    
                for j in range(len(myinfrastructure.base_stations)):    
            
                    if i==j:
                        continue
            
                    thevars = []
                    thecoefs = []          
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):  
                        thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                        thecoefs.append(1.0)             
                        
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[j].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):      
                        thevars.append(x[myindex[str(tau) + "." + str(j) + "." + str(d) + "." + str(p)]])
                        thecoefs.append(-1.0)                           
                    
                    cName = "c5." + str(tau) + "." + str(i) + "." + str(d) + "." + str(i)  + "." + str(j)
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0], names=[cName])  
       
    ## 6. Delay constraint
     
    max_delay = 0.0
    
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
        
            thevars = []
            thecoefs = []         
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):    
                    
                    
                    thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0*float(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['latency'][p]))
                    #print("{}: delay from bs {} to cu {} is {}".format(p, myinfrastructure.base_stations[i].idx, myinfrastructure.dcs[d].idx, myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['latency'][p]))
                    #print("Requested delay is {}".format(float(mytenants[tau].request.delay)))
    
                    
                    cName = "c6." + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[float(mytenants[tau].request.delay)], names=[cName])

                    
                    #print(cpx.linear_constraints.get_rows(cName)
                    
    
    
    # 7. coupled constraint 1: not more bitrate than requested or nothing and exclude normal DC from low-delay tenants
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):     
                    
                    thevars = []
                    thecoefs = []  

                    thevars.append(z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0)
                    
                    thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(-1.0*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]))     
                    
                    cName = "c7." + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0], names=[cName])         
        
    # 8. coupled constraint 2: at least forecasted bitrate or nothing
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(-1.0)
                    
                    thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)])                   
                    #thecoefs.append(1.0)    
                    
                    cName = "c8." + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0], names=[cName])   
            

                    
    ## 9. auxiliary constraint 1 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0)
                    
                    thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(-1.0*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]))           
                    
                    cName = "c9." + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0], names=[cName])       
                    
                    
    ## 10. auxiliary constraint 2 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0)
                    
                    thevars.append(z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(-1.0)                   
                    
                    cName = "c10." + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0], names=[cName])              
                    
    ## 11. auxiliary constraint 3 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(-1.0)
                    
                    thevars.append(z[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(1.0)                    
                    
                    thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                    thecoefs.append(float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]))             
                    
                    cName = "c11." + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])], names=[cName])     
                    
                    
    ## 12. force previously accepted tenants to remain accepted if its duration has not expired to avoid ping pong effect 
    for tau in range(len(mytenants)):
        if int(mytenants[tau].accepted) == 1:
            if int(mytenants[tau].request.duration) > 0:
                print("!!!!!!!!!!!!tau {} must be accepted, duration={}".format(tau,mytenants[tau].request.duration))
                #time.sleep(5)
                #this tenant has to be accepted
                thevars = []
                thecoefs = []        
                for i in range(len(myinfrastructure.base_stations)):
                    for d in range(len(myinfrastructure.dcs)):
                        for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'])):                 
                            thevars.append(x[myindex[str(tau) + "." + str(i) + "." + str(d) + "." + str(p)]])
                            thecoefs.append(-1.0)
                
                cName = "c12." + str(tau) 
                cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                    senses=["L"], rhs=[-1.0], names=[cName])                
            
    return myindex
        

def solve(mytenants, myinfrastructure):
    #print("Solver")
    
    print("Building model...")
    start = time.time()
    z = [] # resource allocation decisions
    x = [] # admission control and routing decisions
    y = [] # auxiliary variable to linearize quadratic objective
    u = [] # additional (leased) CU capacity
    v = [] # additional (leased) BS capacity
    w = [] # additional (leased) transport link capacity

    myprob = cplex.Cplex()
    myindex = createMILP(myprob, mytenants, myinfrastructure, z, x, y, u, v, w)
    
    end = time.time()
    elapsed = (end - start)
    print("DONE (in {0} secs)".format(elapsed))
    #f = open(ResultFile, 'a')
    #f.write(str(elapsed) + "\t")
    #f.close()
    
    print("Starting Solver....")

    #myprob.set_log_stream(None)
    #master.set_error_stream(None)
    #master.set_warning_stream(None)
    #myprob.set_results_stream(None)
    
    myprob.parameters.timelimit.set(5000)
    myprob.parameters.mip.tolerances.mipgap.set(0.05)    

    start = time.time()
    myprob.solve()
    
    end = time.time()
    elapsed = (end - start)
    solution = myprob.solution
    
    #f = open(ResultFile, 'a')
    #f.write(str(elapsed) + "\t")
    #f.close()
        

    if solution.get_status() == solution.status.MIP_optimal or solution.get_status() == solution.status.optimal_tolerance or solution.get_status() == solution.status.MIP_time_limit_feasible:
        print("FINISHED (status {0}) in {1} secs".format(solution.get_status(), elapsed))    
        print("Solution status: ", solution.get_status())
        print("Objective value: ", solution.get_objective_value())          
        
        
        zsol = solution.get_values(z)
        xsol = solution.get_values(x)
        usol = solution.get_values(u)
        vsol = solution.get_values(v)
        wsol = solution.get_values(w)
        
        return zsol, xsol, usol, vsol, wsol, myindex
                        
    else:
        print("Solution not solved (this should not happen!): {0}".format(solution.get_status()))
        
        variable = raw_input('Please sth to continue')
        return -1
    
