#!/usr/bin/python
import sys
sys.path.append('../helpers/')

import os
import numpy as np
import math
from helpers import *
import time
#import matplotlib.pyplot as plt
max_history = 10*mon_samples

it_counter = 0
agg_reward = 0
agg_penalty = 0

#raw_input("Press key to continue...")
#0.1) Topology

topo_file = os.path.join('..', 'examples', 'enso_topo')
G = nx.Graph()
mynetwork = Network(G)
base_station = []
dc = []
myinfrastructure = Infrastructure(mynetwork, base_station, dc)
myinfrastructure.fromJSON(topo_file)
NumBS = len(myinfrastructure.base_stations)

#0.2) Tenant requests
alltenants = AllTenants()
alltenants.fromJSON(tenants_file)
mytenants = alltenants.tenant

NumTenants = len(mytenants)

tenantsResultFile = "/tmp/enso_tenants_result.json"
rewardPenaltyFile = "/tmp/enso_penalty_reward.json"
muxGainFile = "/tmp/enso_mux_gain.json"

print("############# Interval = {} Generating... ####################".format(alltenants.decision_interval))


#print("Time = {}".format(t))
for tau in range(0, NumTenants):
	if mytenants[tau].accepted == 0:
		continue
	#print("Tenant {0}: ".format(mytenants[tau].idx))
	for i in range( 0, NumBS):
		Lambda = mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]
		lambda_i = mytenants[tau].service.mean_bitrate[str(myinfrastructure.base_stations[i].idx)]
		sigma_i = mytenants[tau].service.std_bitrate[str(myinfrastructure.base_stations[i].idx)]


		if len(mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)]) > 0:
			Traffic = [ mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)][str(k)] for k in range(len(mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)])) ] # get history data in order            
		else:
			Traffic = []

		for t in range(mon_samples): # simulate traffic

			sample = min( float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]), max(0, np.random.normal( float(mytenants[tau].service.mean_bitrate[str(myinfrastructure.base_stations[i].idx)]), float(mytenants[tau].service.std_bitrate[str(myinfrastructure.base_stations[i].idx)]))))
			if len(Traffic) >= max_history: # shift window
				Traffic = Traffic[1:]
			Traffic.append(sample) #add sample


		mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)] = dict(zip([str(e) for e in range(len(Traffic))], Traffic))

	#print("BS {0}, new sample = {1} Mb/s".format(myinfrastructure.base_stations[i].idx, sample))


#plotting
print("time\ttau\tduration\tbs_id\tthr\tmux.g\tpenalty")
#----------------Open the files to write the results----------
pTenantsResultFile = open(tenantsResultFile, "w")
pRewardPenaltyFile = open(rewardPenaltyFile, "w")
pMuxGainFile = open(muxGainFile, "w")

pTenantsResultFile.write("{ \"results\": [")
pRewardPenaltyFile.write("{ \"results\": [")
pMuxGainFile.write("{ \"results\": [")

agg_ran = 0
mux_gain = 0
for t in range(mon_samples): # simulate traffic
	it_counter = it_counter + 1
	agg_penalty = 0
	agg_reward = 0
	for tau in range(0, NumTenants):
		print("tau={}, accpeted={}".format(tau, mytenants[tau].accepted))
		if mytenants[tau].accepted == 0:
			continue

		for i in range( 0, NumBS):
			agg_ran = agg_ran + mytenants[tau].nw_alloc[str(myinfrastructure.base_stations[i].idx)]

			Traffic = [ mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)][str(k)] for k in range(len(mytenants[tau].data_history[str(myinfrastructure.base_stations[i].idx)])) ] # get history data in order
			current_traffic = Traffic[-mon_samples+t]
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
			penalty = max(0, float(current_traffic) - float(mytenants[tau].nw_alloc[str(myinfrastructure.base_stations[i].idx)]))

			print("{0}\t{1}\t{2}\t{3}\t{4}\t{5:0.01f}\t{6}\t{7}\t{8}\t{9}\t{10}".format(t, mytenants[tau].idx, mytenants[tau].request.duration, myinfrastructure.base_stations[i].idx, float(current_traffic), mux_gain, penalty, float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]), mytenants[tau].nw_alloc[str(myinfrastructure.base_stations[i].idx)], agg_request, agg_reserved))

			#TBD: ESTO ES PARA EL SERVIDOR QUE PLOTEE: HTTP POST AL CONTROLADOR PRINCIPAL

			#print("\{\"time\": {0}, \"mytenants[tau].idx\": {1} \"mytenants[tau].request.duration\": {2}, \"myinfrastructure.base_stations[i].idx\":{3} \"float(current_traffic)\": {4}, \"mux_gain\": {5:0.01f}, \"penalty\": {6}, \"float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)])\": {7}, \"mytenants[tau].nw_alloc[str(myinfrastructure.base_stations[i].idx)]\": {8}, \"agg_request\": {9}, \"agg_reserved\": {10}\}".format(t, mytenants[tau].idx, mytenants[tau].request.duration, myinfrastructure.base_stations[i].idx, float(current_traffic), mux_gain, penalty, float(mytenants[tau].request.bitrate[str(myinfrastructure.base_stations[i].idx)]), mytenants[tau].nw_alloc[str(myinfrastructure.base_stations[i].idx)], agg_request, agg_reserved))    

			pTenantsResultFile.write("{{\"time\": {0}, \"tenant\": {1}, \"baseStation\":{2}, \"traffic\": {3:.2f} }},".format(t, mytenants[tau].idx, myinfrastructure.base_stations[i].idx, float(current_traffic)))

			agg_penalty = agg_penalty + float(mytenants[tau].penalty)*penalty


		agg_reward = agg_reward + float(mytenants[tau].reward)


	pRewardPenaltyFile.write("{{\"time\": {0}, \"aggReward\": {1:.2f}, \"aggPenalty\": {2:.6f}}},".format(t, (agg_reward), (agg_penalty)))
	#time.sleep(1)
	print('----------------------------------------------------')
#TBD Guardar agg_penalty agg_reward
#reward_penalty_file.write("{{\"time\": {0}, \"aggReward\": {1}, \"aggPenalty\": {2}}},".format(t, agg_reward, agg_penalty));


#Write the penalty and multiplexing gain

#for t in range(mon_samples):
#    pRewardPenaltyFile.write("{{\"time\": {0}, \"aggReward\": {1:.2f}, \"aggPenalty\": {2:.6f}}},".format(t, (agg_reward / mon_samples), (agg_penalty / mon_samples)))

for t in range(mon_samples):
    pMuxGainFile.write("{{\"time\": {0}, \"muxGain\": {1:.2f}}},".format(t, mux_gain))

print('SUMMARY')
print('iteration={0}\tmean reward={1:0.1f}\tmean penalty={2:0.1f}\tmean net gain={3:0.1f}\tagg RAN capacity={4:.01f}'.format(it_counter, agg_reward/it_counter, agg_penalty/it_counter, (agg_reward-agg_penalty)/it_counter, agg_ran/it_counter))
print('----------------------------------------------------')

numAcceptedTenants = 0

for tau in range(0, NumTenants):
    if mytenants[tau].accepted != 0:
        numAcceptedTenants = numAcceptedTenants + 1

if(numAcceptedTenants > 0):
    pTenantsResultFile.seek(-1, 2)
    pRewardPenaltyFile.seek(-1, 2)
    pMuxGainFile.seek(-1, 2)
else:
    pTenantsResultFile.seek(0, 2)
    pRewardPenaltyFile.seek(0, 2)
    pMuxGainFile.seek(0, 2)

pTenantsResultFile.write("]}")
pTenantsResultFile.close()

pRewardPenaltyFile.write("]}")
pRewardPenaltyFile.close()

pMuxGainFile.write("]}")
pMuxGainFile.close()

# write tenant file                
alltenants.toJSON(tenants_file)






