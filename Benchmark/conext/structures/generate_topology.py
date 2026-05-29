import sys 
sys.path.append('../helpers/')
import networkx as nx
from helpers import *

def generate_onelink_topology(topo_file, capacity = 1):

    print("############# Generating simplest network.. ####################")

    this_cap = capacity
    
    ## 1) Scenario -- TBD: this should come from the orchestrator....  ##
    G = nx.Graph()
    G.add_nodes_from([1,2])
    G.add_edges_from([(1, 2, {'cap' : this_cap, 'delay' : 0} )])

    ## 1.1) Infrastructure ##
    mynetwork = Network(G)

    base_station = [BaseStation(1, this_cap)] # graph node id, capacity
    dc           = [DC(2, this_cap)] # graph node id, lowdelay={1,0}

    myinfrastructure = Infrastructure(mynetwork, base_station, dc)

    print("{} BSs, capacity: {}, {} DCs, {} Nodes, {} links".format(len(base_station), 
          this_cap, len(dc), len(G.nodes()), len(G.edges())))
    print("dc, capacity: {}".format(dc[0].capacity))


    ## 1) Find all paths ##

    print("Finding all paths between bs and dc...")
    start = time.time()
    mypaths = [] #tau, i, d

    for i in range(len(myinfrastructure.base_stations)):
        mypaths_i = []
        for d in range(len(myinfrastructure.dcs)):
            mypaths_i_d = []
            for p in nx.all_simple_paths(myinfrastructure.network.graph, myinfrastructure.base_stations[i].idx, myinfrastructure.dcs[d].idx):
                mypaths_i_d.append(p)
            #mypaths2d.append([p for p in nx.all_simple_paths(myinfrastructure.network.graph, myinfrastructure.base_stations[i].idx, myinfrastructure.dcs[d].idx)])
            mypaths_i.append(mypaths_i_d)
        mypaths.append(mypaths_i)
    end = time.time()
    elapsed = (end - start)
    print("DONE (in {0} secs)".format(elapsed))       


    for i in range(len(myinfrastructure.base_stations)):
        myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)] = {}
        for d in range(len(myinfrastructure.dcs)):
            myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)] = {}
            myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'] = []
            myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['latency'] = []
            for p in range(len(mypaths[i][d])):
                myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'].append(mypaths[i][d][p])
                #print(myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['path'][(p)])
                mydelay = 0
                for l in range(len(mypaths[i][d][p])-1):
                    #mydelay = mydelay + G.edges()
                    mydelay = mydelay + G[mypaths[i][d][p][l]][mypaths[i][d][p][l+1]]['delay']
                myinfrastructure.paths[str(myinfrastructure.base_stations[i].idx)][str(myinfrastructure.dcs[d].idx)]['latency'].append(mydelay)

    # write topology file
    data = myinfrastructure.toJSON(topo_file)

#%% create file
# capacity = 1
# generate_onelink_topology(os.path.join('.','oneLink_topo.json'), capacity)

