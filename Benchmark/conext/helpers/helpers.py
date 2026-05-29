import sys, os, json
from subprocess import call
from networkx.readwrite import json_graph
import networkx as nx
import time
import errno
# from enso_conf import *

from heapq import heappush, heappop
from itertools import count


# DEBUG = 1
# mon_samples = 12 # Monitoring samples in one decision interval

# monitoring_samples = 30  # Monitoring samples in one decision interval
# n_preds            = 4
# slen               = 24*60/monitoring_samples # season num of samples

class Request:
    def __init__(self, bitrate, delay, duration):
        self.bitrate = bitrate
        self.delay = delay
        self.duration = duration 
        self.duration_remaining = duration 
        self.loop = []
        
class Service:
    def __init__(self, mean_bitrate, std_bitrate, alpha, beta):
        self.mean_bitrate = mean_bitrate
        self.std_bitrate = std_bitrate
        self.alpha = alpha
        self.beta = beta

class Tenant:
    def __init__(self, idx, penalty, reward, request, type_tenant):
        self.idx = idx
        self.penalty = penalty
        self.reward = reward
        self.request = request
        self.forecast_usage = {}
        self.forecast_uncertainty = {}
        self.accepted = 0
        self.nw_alloc = {}
        self.path_alloc = {}
        self.dc_alloc = []
        self.data_history = {}
        self.last_accepted = -1
        self.service = {}
        self.status = "PENDING"
        self.type_tenant = type_tenant
         
 
class AllTenants:
    def __init__(self):
        self.tenant = []
        self.decision_interval = 0
        
    def toJSON(self, myfile):
        def tenant2json(ten):
            return {
            'accepted'    : ten.accepted,
            'data_history': ten.data_history,
            'data_future' : ten.data_future,
            'dc_alloc'    : ten.dc_alloc,
            'forecast_usage' : ten.forecast_usage,
            'forecast_uncertainty' : ten.forecast_uncertainty,
            'idx'         : ten.idx,
            'nw_alloc'    : ten.nw_alloc,
            'path_alloc'  : ten.path_alloc,
            'penalty'     : ten.penalty,
            'request'     : ten.request.__dict__,
            'reward'      : ten.reward,
            'service'     : ten.service.__dict__,
            'type_tenant' : ten.type_tenant,
            'last_accepted': ten.last_accepted,
            'status'      : ten.status
            }            
        with open(myfile, 'w') as outfile:
            json_data = {"tenant": list(map(lambda x: tenant2json(x), self.tenant)), 'decision_interval':self.decision_interval} 
            json.dump(json_data, outfile, sort_keys=True, indent=4)
            # json.dump(self, outfile, default=lambda o: o.__dict__, sort_keys=True, indent=4)    
            # outfile.close()
        
                
    def fromJSON(self, myfile):
        try:
            data = json.load(open(myfile))
            for x in data['tenant']:
                mytenant = Tenant(x['idx'], x['penalty'], x['reward'], Request(x['request']['bitrate'], x['request']['delay'], x['request']['duration']), x['type_tenant'])
                mytenant.request.duration_remaining = x['request']['duration_remaining']
                mytenant.request.loop = x['request']['loop']
                mytenant.data_history = x['data_history']
                mytenant.data_future = x['data_future']
                mytenant.forecast_usage = x['forecast_usage']
                mytenant.accepted = x['accepted']
                mytenant.status = x['status']
                mytenant.nw_alloc = x['nw_alloc']
                mytenant.path_alloc = x['path_alloc']
                mytenant.dc_alloc = x['dc_alloc']

                mytenant.last_accepted = x['last_accepted']
                    
                mytenant.forecast_uncertainty = x['forecast_uncertainty']            
                mytenant.service = Service(x['service']['mean_bitrate'], x['service']['std_bitrate'], x['service']['alpha'], x['service']['beta'])
                self.tenant.append(mytenant)
            self.decision_interval = data['decision_interval']
        except: 
            print(f'Error loading the tenants from file: "{myfile}"')
            self = AllTenants()
            

class Network: # special nodes are BSs and DCs
    def __init__(self, graph):
        self.graph = graph
        
class BaseStation:
    def __init__(self, idx, capacity):
        self.idx = idx    # id in network graph
        self.capacity = capacity
         
class DC:
    def __init__(self, idx, capacity, lowdelay=0):
        self.idx = idx  # id in network graph
        self.capacity = capacity
        self.lowdelay = lowdelay
        # TBD: we should add computational/memory capacities...        
        
class Infrastructure:
    def __init__(self, network = None, base_stations = [], dcs = []):
        self.network = network
        self.base_stations = base_stations
        self.dcs = dcs
        self.paths = {}
        
    def toJSON(self, myfile):
        json_data = {
            'Network' : json_graph.node_link_data(self.network.graph),
            'BaseStations' : list(map(lambda x: x.__dict__, self.base_stations)),
            'DCs' : list(map(lambda x: x.__dict__, self.dcs)),
            'paths' : self.paths
            }
    
        with open(myfile, 'w') as outfile:
            json.dump(json_data, outfile, sort_keys=True, indent=4)

    
    def fromJSON(self, myfile): 
        data = json.load(open(myfile))
        for x in data['BaseStations']: 
            self.base_stations.append(BaseStation(x['idx'], x['capacity']))
        for x in data['DCs']: 
            self.dcs.append(DC(x['idx'], x['capacity'], x['lowdelay']))     
        self.network = Network(json_graph.node_link_graph(data['Network']))
        
        self.paths = data['paths']
             
             
        
def IsLinkInPath(mylink, path):
    for n in range(len(path)-1):
        link = (path[n], path[n+1])
        if link == mylink:
            return True
        
        link = (path[n+1], path[n]) # bidirectional links
        if link == mylink:
            return True    
    return False



## file locking functionaly (based on the one from https://github.com/dmfrey/FileLock)
 
class FileLockException(Exception):
    pass
 
class FileLock(object):
    """ A file locking mechanism that has context-manager support so 
        you can use it in a with statement. This should be relatively cross
        compatible as it doesn't rely on msvcrt or fcntl for the locking.
    """
 
    def __init__(self, file_name, timeout=2, delay=.05):
        """ Prepare the file locker. Specify the file to lock and optionally
            the maximum timeout and the delay between each attempt to lock.
        """
        if timeout is not None and delay is None:
            raise ValueError("If timeout is not None, then delay must not be None.")
        self.is_locked = False
        self.lockfile = os.path.join(os.getcwd(), "%s.lock" % file_name)
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay
 
 
    def acquire(self):
        """ Acquire the lock, if possible. If the lock is in use, it check again
            every `wait` seconds. It does this until it either gets the lock or
            exceeds `timeout` number of seconds, in which case it throws 
            an exception.
        """
        start_time = time.time()
        while True:

            if(not os.path.isfile(self.lockfile)): 
                
                outfile = open(self.lockfile, "w")
                outfile.write("locked")
                outfile.close()
                self.is_locked = True #moved to ensure tag only when locked
                break;
            else:
                
                time.sleep(self.delay)
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException("Timeout occured.")
                
 
    def release(self):
        """ Get rid of the lock by deleting the lockfile. 
            When working in a `with` statement, this gets automatically 
            called at the end.
        """
        if self.is_locked:
            os.unlink(self.lockfile)
            self.is_locked = False
 
 

def k_shortest_paths(G, source, target, k=1, weight='weight'):
    """Returns the k-shortest paths from source to target in a weighted graph G.
    Parameters
    ----------
    G : NetworkX graph
    source : node
       Starting node
    target : node
       Ending node
       
    k : integer, optional (default=1)
        The number of shortest paths to find
    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight
    Returns
    -------
    lengths, paths : lists
       Returns a tuple with two lists.
       The first list stores the length of each k-shortest path.
       The second list stores each k-shortest path.  
    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.
    Examples
    --------
    >>> G=nx.complete_graph(5)    
    >>> print(k_shortest_paths(G, 0, 4, 4))
    ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])
    Notes
    ------
    Edge weight attributes must be numerical and non-negative.
    Distances are calculated as sums of weighted edges traversed.
    """
    if source == target:
        return ([0], [[source]]) 
       
    length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
    if target not in length:
        raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
        
    lengths = [length[target]]
    paths = [path[target]]
    c = count()        
    B = []                        
    G_original = G.copy()    
    
    for i in range(1, k):
        for j in range(len(paths[-1]) - 1):            
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]
            
            edges_removed = []
            for c_path in paths:
                if len(c_path) > j and root_path == c_path[:j + 1]:
                    u = c_path[j]
                    v = c_path[j + 1]
                    if G.has_edge(u, v):
                        edge_attr = G.edge[u][v]
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))
            
            for n in range(len(root_path) - 1):
                node = root_path[n]
                # out-edges
                for u, v, edge_attr in G.edges_iter(node, data=True):
                    G.remove_edge(u, v)
                    edges_removed.append((u, v, edge_attr))
                
                if G.is_directed():
                    # in-edges
                    for u, v, edge_attr in G.in_edges_iter(node, data=True):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))
            
            spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)            
            if target in spur_path and spur_path[target]:
                total_path = root_path[:-1] + spur_path[target]
                total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length[target]                
                heappush(B, (total_path_length, next(c), total_path))
                
            for e in edges_removed:
                u, v, edge_attr = e
                G.add_edge(u, v, edge_attr)
                       
        if B:
            (l, _, p) = heappop(B)        
            lengths.append(l)
            paths.append(p)
        else:
            break
    
    return (lengths, paths)

def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            
            length += G.edge[u][v].get(weight, 1)
    
    return length        





