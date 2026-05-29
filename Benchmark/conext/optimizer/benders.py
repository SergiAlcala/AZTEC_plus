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

#from scipy.stats import genpareto
#from scipy.stats import pareto
#import matplotlib.pyplot as plt

#from decorator import append
import scipy.io as io
import networkx as nx
from helpers import *

benders_iterations = 0


class BendersLazyConsCallback(LazyConstraintCallback):
    def __call__(self):
        x = self.x
        theta = self.theta
        global benders_iterations
        global lower_bound
        benders_iterations = benders_iterations + 1
        
        workerLP = self.workerLP
        myinfrastructure = self.myinfrastructure
        mytenants = self.mytenants
        master = self.master
        

        # Get the current x solution
        x_sol = self.get_values(x)

        # Benders' cut separation
        theta_sol = self.get_values(theta)
        
        #print(x_sol)
        #print(y_sol)
        #print(theta_sol)
        cuts_found, duals = workerLP.separate(x_sol, theta_sol, x, theta)
        
        if(cuts_found):
            #print(workerLP.cutLhs)
            #print(workerLP.cutRhs)

            self.duals = duals
            self.add(constraint=workerLP.cutLhs,
                        sense="L",
                        rhs=workerLP.cutRhs)
        


# This class builds the worker LP (i.e., the dual of flow constraints and
# capacity constraints of the flow MILP) and allows to separate violated
# Benders' cuts.
class createSubLP:

    def __init__(self, cpx, mytenants, myinfrastructure):
        #print("createSubLP()")
        
        cpx.parameters.preprocessing.reduce.set(0)
        cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.primal)
        cpx.objective.set_sense(cpx.objective.sense.maximize)
                
        
        ### Create variables and objective linear coefficients
        mu1 = []
        alpha = 0.0
        for tau in range(len(mytenants)):
            alpha = alpha + mytenants[tau].service.alpha
            
        for c in range(len(myinfrastructure.dcs)):            
            varname = "mu1." + str(c)
            mu1.append(cpx.variables.get_num())
            cpx.variables.add(obj=[alpha-myinfrastructure.dcs[c].capacity],
                              lb=[0.0],
                              ub=[cplex.infinity],
                              names=[varname])

        mu2 = []
        nlinks = myinfrastructure.network.graph.number_of_edges()
        linkid=0
        for link_ in myinfrastructure.network.graph.edges_iter(data=True):

            link = (link_[0], link_[1])
            link_att = link_[2]
                
            #print("link {0} has capacity {1}".format(link,link_att['cap']))
                
            varname = "mu2." + str(linkid)
            mu2.append(cpx.variables.get_num())
            cpx.variables.add(obj=[-1.0*link_att['cap']],
                            lb=[0.0],
                            ub=[cplex.infinity],
                            names=[varname])        
            linkid = linkid + 1   
                
        mu3 = []
        for b in range(len(myinfrastructure.base_stations)):
            
            varname = "mu3." + str(b)
            mu3.append(cpx.variables.get_num())
            cpx.variables.add(obj=[-1.0*myinfrastructure.base_stations[b].capacity],
                              lb=[0.0],
                              ub=[cplex.infinity],
                              names=[varname])                
            
            
        myindex={}
        idx = 0
        mu4 = []
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        
                        varname = "mu4."  + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                        mu4.append(cpx.variables.get_num())
                        myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)] = idx # map to access variables later   
                        idx = idx + 1
                        cpx.variables.add(obj=[0.0], # it will be updated on each benders iteration with the current x
                                        lb=[0.0],
                                        ub=[cplex.infinity],
                                        names=[varname])                 
                        
        mu5 = []
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        
                        varname = "mu5."  + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                        mu5.append(cpx.variables.get_num())
                        cpx.variables.add(obj=[0.0], # it will be updated on each benders iteration with the current x
                                        lb=[0.0],
                                        ub=[cplex.infinity],
                                        names=[varname])       
                        
                        
        mu6 = []
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        
                        varname = "mu6."  + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                        mu6.append(cpx.variables.get_num())
                        cpx.variables.add(obj=[0.0], # it will be updated on each benders iteration with the current x
                                        lb=[0.0],
                                        ub=[cplex.infinity],
                                        names=[varname])    
                        
                        
        mu7 = []
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        
                        varname = "mu7."  + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                        mu7.append(cpx.variables.get_num())
                        cpx.variables.add(obj=[0.0], # it will be updated on each benders iteration with the current x
                                        lb=[0.0],
                                        ub=[cplex.infinity],
                                        names=[varname])     
                        
        mu8 = []
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        
                        varname = "mu8."  + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                        mu8.append(cpx.variables.get_num())
                        cpx.variables.add(obj=[0.0], # it will be updated on each benders iteration with the current x
                                        lb=[0.0],
                                        ub=[cplex.infinity],
                                        names=[varname])                                         
                
                
        ### Constraints
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        thevars = []
                        thecoefs = []                
                            
                        thevars.append(mu1[c])
                        thecoefs.append(-1.0*mytenants[tau].service.beta)
                        
                        linkid = 0
                        eta = 1.0
                        for link_ in myinfrastructure.network.graph.edges_iter(data=True):        
                            link = (link_[0], link_[1])
                            if IsLinkInPath(link, myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'][p]):
                                thevars.append(mu2[linkid])
                                thecoefs.append(-1.0*eta)
                            linkid = linkid + 1 
                        
                        
                        eta = 1.0
                        thevars.append(mu3[b])
                        thecoefs.append(-1.0*eta)    
                        
                        
                        thevars.append(mu4[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(-1.0)
                        
                        thevars.append(mu5[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(+1.0)
                        
                        #thevars.append(mu6[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        #thecoefs.append(0.0)
                        
                        thevars.append(mu7[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(+1.0)
                        
                        thevars.append(mu8[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(-1.0)                        
                        
                        cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                                senses=["L"],
                                rhs=[0.0])        
                        
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        thevars = []
                        thecoefs = []                
                        
                        thevars.append(mu6[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(-1.0)
                        
                        thevars.append(mu7[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(-1.0)
                        
                        thevars.append(mu8[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(+1.0)         
                        
                        uncertainty = mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[b].idx)]/float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])
                        
                        rho_hat = (mytenants[tau].penalty*uncertainty*mytenants[tau].request.duration)/(float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) - mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)]+ sys.float_info.epsilon + 1)
                          
                        cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                                senses=["L"],
                                rhs=[-1.0*float(rho_hat)])                            

                    
        self.cpx = cpx
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.mu4 = mu4
        self.mu5 = mu5
        self.mu6 = mu6
        self.mu7 = mu7
        self.mu8 = mu8
        self.myinfrastructure = myinfrastructure
        self.mytenants = mytenants
        self.myindex = myindex
        
        self.cut_lhs = None
        self.cut_rhs = None            
                    
    def separate(self, x_sol, theta_sol, x, theta):
        # modify obj function on each benders iteration (constraints do not depend on the master's x solution)
        cpx = self.cpx    
        mu1 = self.mu1
        mu2 = self.mu2
        mu3 = self.mu3
        mu4 = self.mu4
        mu5 = self.mu5
        mu6 = self.mu6
        mu7 = self.mu7
        mu8 = self.mu8
        myindex = self.myindex
        myinfrastructure = self.myinfrastructure
        mytenants = self.mytenants
        CutFound = False
        
        thevars = []
        thecoefs = []
        
        print("[SLAVE] trying with {} accepted tenants...".format(sum(x_sol)/len(myinfrastructure.base_stations)))
        
        alpha = 0.0
        for tau in range(len(mytenants)):
            alpha = alpha + mytenants[tau].service.alpha
            
        for c in range(len(myinfrastructure.dcs)):            
            thevars.append(mu1[c])
            thecoefs.append(alpha-1.0*myinfrastructure.dcs[c].capacity)
            
        linkid=0
        for link_ in myinfrastructure.network.graph.edges_iter(data=True):

            link_att = link_[2]
            thevars.append(mu2[linkid])
            thecoefs.append(-1.0*link_att['cap'])
            
            linkid = linkid + 1   
                
        for b in range(len(myinfrastructure.base_stations)):
            
            thevars.append(mu3[b])
            thecoefs.append(-1.0*myinfrastructure.base_stations[b].capacity)
            
            
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):               
                        thevars.append(mu4[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(-1.0*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])*x_sol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        
                        
                        thevars.append(mu5[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(float(mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)])*x_sol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        
                        thevars.append(mu6[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(-1.0*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])*x_sol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        
                        thevars.append(mu8[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])*x_sol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]] - float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]))                        
        

        cpx.objective.set_linear(zip(thevars, thecoefs))        
        
        cpx.set_log_stream(None)
        #master.set_error_stream(None)
        #master.set_warning_stream(None)
        cpx.set_results_stream(None)        
        
        start = time.time()
        cpx.solve()
        duals = cpx.solution.get_dual_values()

        #print("!!!!!!!!!!!!!! DUAL VALUES = {}".format(duals))
        end = time.time()

        if cpx.solution.get_status() == cpx.solution.status.unbounded:
            print("dual slave is unbounded! --- add feasibility cut")

            # Get the violated cut as an unbounded ray of the worker LP
            ray = cpx.solution.advanced.get_ray()
            
            #Compute the cut from the unbounded ray.
            cutVarsList = []
            cutCoefsList = []  
            for tau in range(len(mytenants)):
                for b in range(len(myinfrastructure.base_stations)):
                    for c in range(len(myinfrastructure.dcs)):
                        for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):   

                            mu4_idx = len(mu1) + len(mu2) + len(mu3) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            mu5_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            mu6_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + len(mu5) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            mu8_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + len(mu5) + len(mu6) + len(mu7) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            
                            thecoef = -1.0*ray[mu4_idx]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])  \
                                      +1.0*ray[mu5_idx]*float(mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)]) \
                                      -1.0*ray[mu6_idx]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) \
                                      +1.0*ray[mu8_idx]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])
                            
                            cutVarsList.append(x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                            cutCoefsList.append(thecoef)
            cutLhs = cplex.SparsePair(ind=cutVarsList, val=cutCoefsList)            
            cutRhs = 0.0
            
            
            alpha = 0.0
            for tau in range(len(mytenants)):
                alpha = alpha + mytenants[tau].service.alpha
            for c in range(len(myinfrastructure.dcs)):
                cutRhs = cutRhs + (float(myinfrastructure.dcs[c].capacity)- alpha)*ray[c]    
                

                
            linkid=0
            for link_ in myinfrastructure.network.graph.edges_iter(data=True):

                link_att = link_[2]                
                           
                cutRhs = cutRhs + float(link_att['cap'])*ray[len(mu1) + linkid]  
                linkid = linkid + 1
                

            for b in range(len(myinfrastructure.base_stations)):
                cutRhs = cutRhs + (float(myinfrastructure.base_stations[b].capacity))*ray[len(mu1) + len(mu2) + b]             
                
            for tau in range(len(mytenants)):
                for b in range(len(myinfrastructure.base_stations)):
                    for c in range(len(myinfrastructure.dcs)):
                        for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])): 
                            mu8_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + len(mu5) + len(mu6) + len(mu7) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            cutRhs = cutRhs + float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])*ray[mu8_idx]
                            
            self.cutLhs = cutLhs
            self.cutRhs = cutRhs
            CutFound = True         
            return CutFound, duals

        elif cpx.solution.get_status() == cpx.solution.status.optimal:
            print("slave is feasible! --- add optimality cut")
            
            sol = cpx.solution.get_values()
            
            cutVarsList = []
            cutCoefsList = []  
            for tau in range(len(mytenants)):
                for b in range(len(myinfrastructure.base_stations)):
                    for c in range(len(myinfrastructure.dcs)):
                        for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):   
                            
                            mu4_idx = len(mu1) + len(mu2) + len(mu3) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            mu5_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            mu6_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + len(mu5) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            mu8_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + len(mu5) + len(mu6) + len(mu7) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            
                            thecoef = -1.0*sol[mu4_idx]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) + \
                                       1.0*sol[mu5_idx]*float(mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)]) + \
                                      -1.0*sol[mu6_idx]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) + \
                                       1.0*sol[mu8_idx]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])
                            
                            cutVarsList.append(x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                            cutCoefsList.append(thecoef)
                        
            cutVarsList.append(theta[0])
            cutCoefsList.append(-1.0)                            
                
            cutLhs = cplex.SparsePair(ind=cutVarsList, val=cutCoefsList)            
            cutRhs = 0.0
            
            
            alpha = 0.0
            for tau in range(len(mytenants)):
                alpha = alpha + mytenants[tau].service.alpha
            for c in range(len(myinfrastructure.dcs)):
                cutRhs = cutRhs + (float(myinfrastructure.dcs[c].capacity)- alpha)*sol[c]    
          

                
            linkid=0
            for link_ in myinfrastructure.network.graph.edges_iter(data=True):

                link_att = link_[2]                
                           
                cutRhs = cutRhs + float(link_att['cap'])*sol[len(mu1) + linkid]  
                linkid = linkid + 1
                

            for b in range(len(myinfrastructure.base_stations)):
                cutRhs = cutRhs + (float(myinfrastructure.base_stations[b].capacity))*sol[len(mu1) + len(mu2) + b]             
         
            for tau in range(len(mytenants)):
                for b in range(len(myinfrastructure.base_stations)):
                    for c in range(len(myinfrastructure.dcs)):
                        for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])): 
                            mu8_idx = len(mu1) + len(mu2) + len(mu3) + len(mu4) + len(mu5) + len(mu6) + len(mu7) + myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]
                            cutRhs = cutRhs + float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])*sol[mu8_idx]
                            
                   
            self.cutLhs = cutLhs
            self.cutRhs = cutRhs
            CutFound = True       
            return CutFound, duals

        else:
            print("Unexpected subproblem solution status")
                    
        return False

def createMasterILP(cpx, mytenants, myinfrastructure, x, theta):
        
    #### Create variables and objective linear coefficients
    cpx.objective.set_sense(cpx.objective.sense.minimize)    

    
    # x: access control and routing      
    myindex = {}
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):                    
                                        
                    uncertainty = mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[b].idx)]/float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])
                    
                    rho_hat = (mytenants[tau].penalty*uncertainty*mytenants[tau].request.duration)/(float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) - mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)]+ sys.float_info.epsilon + 1)
                       
                    varName = "x" + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    x.append(cpx.variables.get_num())
                    myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)] = cpx.variables.get_num() # map to access variables later                     
                    cpx.variables.add(obj = [ rho_hat \
                                              * float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) \
                                              - float(mytenants[tau].reward) ], 
                                lb=[0], ub=[1], types=["B"],
                                names=[varName])      
                    
    # theta: surrogate flow cost variable theta
    theta.append(cpx.variables.get_num())
    cpx.variables.add(obj = [1.0],
                    lb=[0], ub=[cplex.infinity], types=["C"],
                    names=["theta"])
    
    
    #print("master objective coefficients: {}".format(cpx.objective.get_linear()))
    #### Create constraints

    
    ## 1. just one path per tenant and BS, and one CU
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
        
            thevars = []
            thecoefs = []         
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):    
                    thevars.append(x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0)
                    
            cName = "c1." + str(tau) + "." + str(b)  
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                senses=["L"], rhs=[1.0], names=[cName])            
            
    ## 2. if accepted, do so in all BSs and use the same DC
    for tau in range(len(mytenants)):
        for c in range(len(myinfrastructure.dcs)):
            for b in range(len(myinfrastructure.base_stations)):    
                for j in range(len(myinfrastructure.base_stations)):    
            
                    if b==j:
                        continue
            
                    thevars = []
                    thecoefs = []          
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        thevars.append(x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(1.0)             
                        
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[j].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):      
                        thevars.append(x[myindex[str(tau) + "." + str(j) + "." + str(c) + "." + str(p)]])
                        thecoefs.append(-1.0)                           
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0], names=[cName])  
       
    ## 3. Delay constraint
     
    max_delay = 0.0
    
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
        
            thevars = []
            thecoefs = []         
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):    
                    
                    
                    thevars.append(x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0*float(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['latency'][p]))
                    #print("{}: delay from bs {} to cu {} is {}".format(p, myinfrastructure.base_stations[b].idx, myinfrastructure.dcs[c].idx, myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['latency'][p]))
                    #print("Requested delay is {}".format(float(mytenants[tau].request.delay)))
    
                    
                    cName = "c3." + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[float(mytenants[tau].request.delay)], names=[cName])

                    
                    #print(cpx.linear_constraints.get_rows(cName)
                
                    
    ## 4. force previously accepted tenants to remain accepted if its duration has not expired to avoid ping pong effect 
    for tau in range(len(mytenants)):
        if int(mytenants[tau].accepted) == 1:
            if int(mytenants[tau].request.duration) > 0:
                print("!!!!!!!!!!!!tau {} must be accepted, duration={}".format(tau,mytenants[tau].request.duration))
                #time.sleep(5)
                #this tenant has to be accepted
                thevars = []
                thecoefs = []        
                for b in range(len(myinfrastructure.base_stations)):
                    for c in range(len(myinfrastructure.dcs)):
                        for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):                 
                            thevars.append(x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                            thecoefs.append(-1.0)
                
                cName = "c4." + str(tau) 
                cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                    senses=["L"], rhs=[-1.0], names=[cName])                
            
    return myindex


def solve_slave(mytenants, myinfrastructure, x):
    
    z = []
    y = []
    cpx = cplex.Cplex()
    #build model
  #### Create variables and objective linear coefficients
    cpx.objective.set_sense(cpx.objective.sense.minimize)    
    
    myindex = {}
    # z: bitrate 
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):
                    
                        
                    # add variable and cost factor associated in the obj function
                    varName = "z" + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    z.append(cpx.variables.get_num())
                    myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)] = cpx.variables.get_num() # map to access variables later 
                    
                    cpx.variables.add(obj = [0.0], 
                                lb=[0.0], ub=[float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])], types=["C"],
                                names=[varName]) 
         
                    
    # y: auxiliary variable to linearize quadratic objective
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):                    
                    
                    varName = "y" + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    y.append(cpx.variables.get_num())
                    
                    uncertainty = mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[b].idx)]/float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])

                    rho_hat = (mytenants[tau].penalty*uncertainty*mytenants[tau].request.duration)/(float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) - mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)]+ sys.float_info.epsilon + 1)
                                       
                    cpx.variables.add(obj = [ -1.0*rho_hat ], 
                                lb=[0], ub=[float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])], types=["C"],
                                names=[varName])              
                    
                    
    ### Constraints
    
    # 1. Capacity constraints on CUs
    
    #beta = 1.0
    #alpha = 0.0

    alpha_all = 0.0
    for c in range(len(myinfrastructure.dcs)):
        thevars = []
        thecoefs = []
        for tau in range(len(mytenants)):
            alpha = mytenants[tau].service.alpha
            beta = mytenants[tau].service.beta
            if (alpha == 0) and (beta == 0):
                continue
            
            alpha_all = alpha_all + alpha
            for b in range(len(myinfrastructure.base_stations)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):    
                    thevars.append(z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0*beta)
                    

        cName = "c1." + str(c) 


        #print(cpx.linear_constraints.get_rows())
        #print(thecoefs)
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[myinfrastructure.dcs[c].capacity - alpha_all], names=[cName])    
        
    ## 2. Capacity constraints on network links    
    
    eta = 1.0
    nlinks = myinfrastructure.network.graph.number_of_edges()
    linkid=0
    for link_ in myinfrastructure.network.graph.edges_iter(data=True):
        thevars = []
        thecoefs = []        
        link = (link_[0], link_[1])
        link_att = link_[2]
        
        unused_link = 1
        for tau in range(len(mytenants)):
            for b in range(len(myinfrastructure.base_stations)):
                for c in range(len(myinfrastructure.dcs)):
                    for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):  
                        if IsLinkInPath(link, myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'][p]):
                            #print(tau)
                            #print(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'][p])
                            if (z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]] in thevars): # TBD we must ensure that eta is the same as well!!!
                                continue
                                
                            thevars.append(z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                            thecoefs.append(1.0*eta)
                            unused_link = 0
                            
        if unused_link == 1:
            continue
                            
                            
        cName = "c2." + str(linkid) 
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[link_att['cap']], names=[cName])     
        
    ## 3. Capacity constraints on BSs
    eta = 1.0
    
    for b in range(len(myinfrastructure.base_stations)):
        
        thevars = []
        thecoefs = []
        for tau in range(len(mytenants)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):    
                    thevars.append(z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0*eta)
                    
        
        cName = "c3." + str(b) 

        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["L"], rhs=[myinfrastructure.base_stations[b].capacity], names=[cName])         
    
  
    # 4. coupled constraint 1: not more bitrate than requested or nothing and exclude normal DC from low-delay tenants
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):     
                    
                    thevars = []
                    thecoefs = []  

                    thevars.append(z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0)
                    
                    
                    cName = "c7." + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])], names=[cName])         
        
    # 5. coupled constraint 2: at least forecasted bitrate or nothing
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(-1.0)
                    
                    cName = "c8." + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[-1.0*x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]]*mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)]], names=[cName])   
            

                    
    ## 6. auxiliary constraint 1 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0)       
                    
                    cName = "c9." + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])], names=[cName])       
                    
                    
    ## 7. auxiliary constraint 2 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0)
                    
                    thevars.append(z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(-1.0)                   
                    
                    cName = "c10." + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[0.0], names=[cName])              
                    
    ## 8. auxiliary constraint 3 (to linearize quadratic objective)
    for tau in range(len(mytenants)):
        for b in range(len(myinfrastructure.base_stations)):
            for c in range(len(myinfrastructure.dcs)):
                for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):                
                    thevars = []
                    thecoefs = []                

                    thevars.append(y[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(-1.0)
                    
                    thevars.append(z[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]])
                    thecoefs.append(1.0)                    
    
                    
                    cName = "c11." + str(tau) + "." + str(b) + "." + str(c) + "." + str(p)
                    
                    cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                        senses=["L"], rhs=[float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)]) - x[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]]*float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[b].idx)])], names=[cName]) 
    
    cpx.set_log_stream(None)
    #master.set_error_stream(None)
    #master.set_warning_stream(None)
    cpx.set_results_stream(None)
    
    cpx.set_problem_type(cpx.problem_type.LP)
    cpx.solve()
  
    solution = cpx.solution
    zsol = solution.get_values(z)
    
    return zsol

def solve(mytenants, myinfrastructure):
    #print("Solver")
    
    print("Building model...")
    start = time.time()
    z = [] # resource allocation decisions
    x = [] # admission control and routing decisions
    y = [] # auxiliary variable to linearize quadratic objective


    
    master = cplex.Cplex()
    theta = [] # surrogate flow cost for master problem
    myindex = createMasterILP(master, mytenants, myinfrastructure, x, theta)    
    
    
    sub = cplex.Cplex()
    workerLP = createSubLP(sub,  mytenants, myinfrastructure)

    end = time.time()
    elapsed = (end - start)
    print("DONE (in {0} secs)".format(elapsed))
    #f = open(ResultFile, 'a')
    #f.write(str(elapsed) + "\t")
    #f.close()
    
    
    global benders_iterations
    
    # attach a Benders callback to the master
    lazyBenders = master.register_callback(BendersLazyConsCallback)
    lazyBenders.x = x    
    lazyBenders.theta = theta
    lazyBenders.duals = []
    lazyBenders.mytenants =  mytenants
    lazyBenders.myinfrastructure = myinfrastructure
    lazyBenders.workerLP = workerLP
    lazyBenders.master = master


    master.parameters.preprocessing.presolve.set(master.parameters.preprocessing.presolve.values.off)
    master.parameters.mip.strategy.search.set(
    master.parameters.mip.strategy.search.values.traditional)
    master.parameters.timelimit.set(10000)
    master.parameters.mip.tolerances.mipgap.set(0.05)
    #master.set_log_stream(None)
    #master.set_error_stream(None)
    #master.set_warning_stream(None)
    #master.set_results_stream(None)

    print("Starting Solver....")

    start = time.time()
    master.solve()
    end = time.time()
    elapsed = (end - start)
    
    
    
    solution = master.solution
    

    
    
    
    #f = open(ResultFile, 'a')
    #f.write(str(elapsed) + "\t")
    #f.close()
        

    if solution.get_status() == solution.status.MIP_optimal or solution.get_status() == solution.status.optimal_tolerance or solution.get_status() == solution.status.MIP_time_limit_feasible:
        print("FINISHED (status {0}) in {1} secs".format(solution.get_status(), elapsed))    
        print("Solution status: ", solution.get_status())
        print("Objective value: ", solution.get_objective_value())          
        
        #f = open(ResultFile, 'a')
        #f.write(str(benders_iterations) + "\t")
        #f.close()    
        
        xsol = solution.get_values(x)
        #print(x)
        
        
        #zsol = []
        
        zsol = solve_slave(mytenants, myinfrastructure, xsol)
        #for tau in range(len(mytenants)):
            #for b in range(len(myinfrastructure.base_stations)):
                #for c in range(len(myinfrastructure.dcs)):
                    #for p in range(len(myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'])):   

                        #zsol.append([])
                        #zsol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]] = lazyBenders.duals[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]]
                        #if zsol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]] > 1e-6 or xsol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]] > 0.5:
                        
                            #print("[tau {0}] BS {1}, DC {2}, path {3}, bitrate = {4:0.01f} Mb/s, x={5} (forecast={6:0.01f} + {7:0.01f} Mb/s, Reward={7}, Penalty={8})".format(tau, b, c, myinfrastructure.paths[str(myinfrastructure.base_stations[b].idx)][str(myinfrastructure.dcs[c].idx)]['path'][p], zsol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]], xsol[myindex[str(tau) + "." + str(b) + "." + str(c) + "." + str(p)]], mytenants[tau].forecast_usage[str(myinfrastructure.base_stations[b].idx)], mytenants[tau].forecast_uncertainty[str(myinfrastructure.base_stations[b].idx)], float(mytenants[tau].reward), float(mytenants[tau].penalty)))
                        
        #print(zsol)

        usol = zeros(len(myinfrastructure.dcs))
        vsol = zeros(len(myinfrastructure.base_stations))
        wsol = zeros(myinfrastructure.network.graph.number_of_edges())
        
        return zsol, xsol, usol, vsol, wsol, myindex
                        
    else:
        print("Solution not solved (this should not happen!): {0}".format(solution.get_status()))
        
        variable = raw_input('Please sth to continue')
        return -1
    
