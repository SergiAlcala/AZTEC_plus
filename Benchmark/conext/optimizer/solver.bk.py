from __future__ import print_function

import sys
sys.path.append('helpers/')
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

        

def createMILP(cpx, paths, mytenants, myinfrastructure, z, r, y):
        
    #### Create variables and objective linear coefficients
    cpx.objective.set_sense(cpx.objective.sense.minimize)    
    
    myindex = {}
    longest_path = 0
    shortest_path = 1e9
    # z: bitrate 
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):
                    
                    # figure out longest and shortest path to fine tune rewards later on
                    if len(paths[i][d][p])>longest_path:
                        longest_path = len(paths[i][d][p])
                        
                    if len(paths[i][d][p])<shortest_path:
                        shortest_path = len(paths[i][d][p])
                        
                    # add variable and cost factor associated in the obj function
                    varName = "z" + str(tau) + str(i) + str(d) + str(p)
                    z.append(cpx.variables.get_num())
                    myindex[str(tau) + str(i) + str(d) + str(p)] = cpx.variables.get_num() # map to access variables later 
                    
                    cpx.variables.add(obj = [0.0], 
                                lb=[0.0], ub=[mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]], types=["C"],
                                names=[varName]) 
    #print("longest path={}, shortest_path={}".format(longest_path, shortest_path))
    
    # r: access control                
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):                    
                    
                    max_path_penalty = 0.001 # relative reward (between 0 and 1) reduction for longest penalty <- This should be small but larger than 0
                    min_path_penalty = 0 # relative reward (between 0 and 1) reduction for shortest penalty (reward reduction on other path lengths are proportional)
                    
                    b = (max_path_penalty - min_path_penalty)/(longest_path - shortest_path) # slope
                    a = min_path_penalty -b*shortest_path # offset
                    
                    mypath_penalty = a + b*len(paths[i][d][p]) # reward is penalized depending on the path length
                    
                    norm_factor = mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)] \
                            - mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)] \
                            +sys.float_info.epsilon
                        
                    risk1 = 1.0/norm_factor
                    risk2 = mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[i].idx)]/norm_factor
                    
                    #print("tau={}, risk = {}".format(tau, risk1*risk2))
                    
                    varName = "r" + str(tau) + str(i) + str(d) + str(p)
                    r.append(cpx.variables.get_num())
                    cpx.variables.add(obj = [ mytenants[tau].penalty*risk1*risk2 \
                                              * mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)] \
                                              - mytenants[tau].reward*(1.0-mypath_penalty) ], 
                                lb=[0], ub=[1], types=["B"],
                                names=[varName])      
                    
    # y: auxiliary variable to linearize quadratic objective
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):                    
                    
                    varName = "y" + str(tau) + str(i) + str(d) + str(p)
                    y.append(cpx.variables.get_num())
                    
                    norm_factor = mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)] \
                            - mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)] \
                            +sys.float_info.epsilon
                        
                    risk1 = 1.0/norm_factor
                    risk2 = mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[i].idx)]/norm_factor
                    
                    cpx.variables.add(obj = [ -1.0*mytenants[tau].penalty*risk1*risk2 ], 
                                lb=[0], ub=[mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]], types=["C"],
                                names=[varName])          
                   
    #print(cpx.objective.get_linear())
    #print(cpx.variables.get_names())
                    
    
    
    #### Create constraints
    
    # 1. Capacity constraints on BSs

    for i in range(len(myinfrastructure.base_stations)):
        
        thevars = []
        thecoefs = []
        for tau in range(len(mytenants)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):    
                    thevars.append(z[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(1.0)

        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[myinfrastructure.base_stations[i].capacity])    
                                    
    
    ## 2. Capacity constraints on network links

    nlinks = myinfrastructure.network.graph.number_of_edges()
    for link_ in myinfrastructure.network.graph.edges_iter(data=True):
        
        thevars = []
        thecoefs = []        
        link = (link_[0], link_[1])
        link_att = link_[2]
        #print("link {0} has capacity {1}".format(link,link_att['cap']))
        for tau in range(len(mytenants)):
            for i in range(len(myinfrastructure.base_stations)):
                for d in range(len(myinfrastructure.dcs)):
                    for p in range(len(paths[i][d])):  
                        #print(paths[i][d][p])
                        if IsLinkInPath(link, paths[i][d][p]):
                            thevars.append(z[myindex[str(tau) + str(i) + str(d) + str(p)]])
                            thecoefs.append(1.0)
                            
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[link_att['cap']])                    

    
    ## 3. just one path per tenant and BS
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
        
            thevars = []
            thecoefs = []         
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):    
                    thevars.append(r[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(1.0)
                    
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                senses=["L"], rhs=[1.0])            
        
    # 4. coupled constraint 1: not more bitrate than requested or nothing and exclude normal DC from low-delay tenants
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):     
                    
                    thevars = []
                    thecoefs = []  

                    thevars.append(z[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(1.0)
                    
                    thevars.append(r[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    if mytenants[tau].request.lowdelay == 1 and myinfrastructure.dcs[d].lowdelay == 0:
                        thecoefs.append(0)
                    else:
                        thecoefs.append(-1.0*mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])                    
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0])         
        
    # 5. coupled constraint 2: at least forecasted bitrate or nothing
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(z[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(-1.0)
                    
                    thevars.append(r[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[i].idx)])                   
                    #thecoefs.append(1.0)                    
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0])   
            
    ## 6. if accepted, do so in all BSs and use the same DC
    for tau in range(len(mytenants)):
        for d in range(len(myinfrastructure.dcs)):
            for i in range(len(myinfrastructure.base_stations)):    
                for j in range(len(myinfrastructure.base_stations)):    
            
                    if i==j:
                        continue
            
                    thevars = []
                    thecoefs = []          
                    for p in range(len(paths[i][d])):  
                        thevars.append(r[myindex[str(tau) + str(i) + str(d) + str(p)]])
                        thecoefs.append(1.0)             
                        
                    for p in range(len(paths[j][d])):      
                        thevars.append(r[myindex[str(tau) + str(j) + str(d) + str(p)]])
                        thecoefs.append(-1.0)                           
                    
                            
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0])  
                    
    ## 7. auxiliary constraint 1 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(1.0)
                    
                    thevars.append(r[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(-1.0*mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])                    
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0])       
                    
                    
    ## 8. auxiliary constraint 2 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(1.0)
                    
                    thevars.append(z[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(-1.0)                    
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0])              
                    
    ## 9. auxiliary constraint 3 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for i in range(len(myinfrastructure.base_stations)):
            for d in range(len(myinfrastructure.dcs)):
                for p in range(len(paths[i][d])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(-1.0)
                    
                    thevars.append(z[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(1.0)                    
                    
                    thevars.append(r[myindex[str(tau) + str(i) + str(d) + str(p)]])
                    thecoefs.append(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])                    
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]])     
                    
                    
    ## 10. force previously accepted tenants to remain accepted if its duration has not expired to avoid ping pong effect 
    #TBD: There is an issue when the forecast of forced accepted tenants exceed capacity: in this case no solution can be found. This can happen even if the tenant does not change its steady state statistics! What should we do? a) extra capacity temporary added b) assigning less than forecast (how?) c) kicking out some tenant (how?)
    #for tau in range(len(mytenants)):
        #if mytenants[tau].accepted == 1:
            #if mytenants[tau].request.duration > 0:
                ##print("tau {} must be accepted, duration={}".format(tau,mytenants[tau].request.duration))
                ##time.sleep(5)
                ##this tenant has to be accepted
                #thevars = []
                #thecoefs = []        
                #for i in range(len(myinfrastructure.base_stations)):
                    #for d in range(len(myinfrastructure.dcs)):
                        #for p in range(len(paths[i][d])):                 
                            #thevars.append(r[myindex[str(tau) + str(i) + str(d) + str(p)]])
                            #thecoefs.append(-1.0)
                
                #cpx.linear_constraints.add(
                    #lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                    #senses=["L"], rhs=[-1.0])                
            
    return myindex
        

def solve(mytenants, myinfrastructure, paths):
    #print("Solver")
    
    print("Building model...")
    start = time.time()
    z = [] # resource allocation decisions
    r = [] # admission control decisions
    y = [] # auxiliary variable to linearize quadratic objective

    myprob = cplex.Cplex()
    myindex = createMILP(myprob, paths, mytenants, myinfrastructure, z, r, y)
    
    end = time.time()
    elapsed = (end - start)
    print("DONE (in {0} secs)".format(elapsed))
    
    print("Starting Solver....")

    myprob.set_log_stream(None)
    #master.set_error_stream(None)
    #master.set_warning_stream(None)
    myprob.set_results_stream(None)

    start = time.time()
    myprob.solve()
    
    end = time.time()
    elapsed = (end - start)
    solution = myprob.solution

    if solution.get_status() == solution.status.MIP_optimal or solution.get_status() == solution.status.optimal_tolerance or solution.get_status() == solution.status.MIP_time_limit_feasible:
        print("FINISHED (status {0}) in {1} secs".format(solution.get_status(), elapsed))    
        print("Solution status: ", solution.get_status())
        print("Objective value: ", solution.get_objective_value())          
        
        
        zsol = solution.get_values(z)
        rsol = solution.get_values(r)
        
        return zsol, rsol, myindex
                        
    else:
        print("Solution not solved (this should not happen -- some tenant has to be kicked out!): {0}".format(solution.get_status()))
        
        time.sleep(100)
        return -1
    
