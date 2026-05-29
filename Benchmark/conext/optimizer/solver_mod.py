from __future__ import print_function

import sys, os
if os.path.join('..','helpers') not in sys.path:
    sys.path.append(os.path.join('..','helpers')) # os.getcwd()

import imp
import cplex
from cplex.callbacks import UserCutCallback, LazyConstraintCallback
from cplex.exceptions import CplexError
from numpy import *
import time


import scipy.io as io
import networkx as nx

from helpers import *
        
def createMILP(cpx, mytenants, myinfrastructure, z, x, y, u, v, w):
    leasing_cost = myinfrastructure.leasing_cost

    #### Create variables and objective linear coefficients
    cpx.objective.set_sense(cpx.objective.sense.minimize)    
    
    myindex = {}

    #### Create variables ############################################

    ### Defining variables z
    for tau in range(len(mytenants)):
        # add variable and cost factor associated in the obj function
        varName = "z" + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
        z.append(cpx.variables.get_num())
        myindex[str(tau)] = cpx.variables.get_num() # map to access variables later 
        
        cpx.variables.add(obj = [0.0], 
                    lb=[0.0], ub=[float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[0].idx)])], 
                    types=["C"], names=[varName])     

    ### Defining variables x an y: access control and routing          
    for tau in range(len(mytenants)):
        # for i in range(len(myinfrastructure.base_stations)):
        # for d in range(len(myinfrastructure.dcs)):
        idx_BS = myinfrastructure.base_stations[0].idx
        idx_CU = myinfrastructure.dcs[0].idx
        
        for p in range(len(myinfrastructure.paths[str(idx_BS)][str(idx_CU)]['path'])):                    
                    
            uncertainty = mytenants[tau].forecast_uncertainty[str(idx_BS)]/float(mytenants[tau].request.bitrate[str(idx_BS)])
            
            print("uncertainty = {}".format(uncertainty))
            #duration = mytenants[tau].request.duration
            duration = 1
            xi_tau_p = uncertainty*duration
            dif_req_real = float(mytenants[tau].request.bitrate[str(idx_BS)]) \
                           - mytenants[tau].forecast_usage[str(idx_BS)] + sys.float_info.epsilon 
            rho_hat  = (mytenants[tau].penalty*xi_tau_p)/dif_req_real 
            
#           print("tau={},bs={} risk = {}, request={}, forecast={}, norm_factor={}".format(mytenants[tau].idx,
#                  i, risk1*risk2, mytenants[tau].request.bitrate[str(idx_BS)], 
#                  mytenants[tau].forecast_usage[str(idx_BS)], norm_factor))

            if uncertainty == 0:
                rho_hat = 0
            
            ### Defining variables x: access control and routing         
            varName = "x" + str(tau) + "." + str(i) + "." + str(d) + "." + str(p)
            x.append(cpx.variables.get_num())
            cpx.variables.add(obj = [ rho_hat \
                                      * float(mytenants[tau].request.bitrate[str(idx_BS)]) \
                                      - float(mytenants[tau].reward)], 
                        lb=[0], ub=[1], types=["B"],
                        names=[varName])      
            
            ### Defining variables y: auxiliary variable to linearize quadratic objective
            varName = "y" + str(tau)  + "." + str(i) + "." + str(d) + "." + str(p)
            y.append(cpx.variables.get_num())
            cpx.variables.add(obj = [ -1.0*rho_hat ], 
                        lb=[0], ub=[float(mytenants[tau].request.bitrate[str(idx_BS)])], types=["C"],
                        names=[varName])          
                    
                     
    ### (eq. 16+) Defining variables u: additional leasing on CU capacity  --> removed
    ### (eq. 16+) Defining variables v: additional leasing on BS capacity  --> removed
    ### (eq. 16+) Defining variables w: additional leasing on network link capacity 
    nlinks = myinfrastructure.network.graph.number_of_edges()
    linkid = 0

    # for link_ in myinfrastructure.network.graph.edges_iter(data=True): # actually a single link...
    for link_ in myinfrastructure.network.graph.edges(data=True): # actually a single link...
        varName = "w" + str(linkid) 
        w.append(cpx.variables.get_num())
        
        cpx.variables.add(obj = [ 1.0*leasing_cost], 
                    lb=[0], types=["C"], 
                    names=[varName])          #ub=[1e6], 
                                    
        linkid = linkid + 1
                   
    #print(cpx.objective.get_linear())
    #print(cpx.variables.get_names())
                    
    cpx.write('logs.lp')    

    #### Create constraints ############################################
    
    # 1. (eq. 2) Capacity constraints on CUs  --> removed

    ## 2. Capacity constraints on network links

    eta = 1.0

    nlinks = myinfrastructure.network.graph.number_of_edges()
    linkid=0
    # for link_ in myinfrastructure.network.graph.edges_iter(data=True): # actually a single link...
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
        
    ## 3. (eq. 4) Capacity constraints on BSs  --> removed
    
    ## 4. (eq. 5) prevent multipath connections --> removed
    
    ## 5. (eq. 6) in all BSs and use same DC --> removed
       
    ## 6. (eq. 7) Delay constraint (eq. 7) --> removed
    
    
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
    
