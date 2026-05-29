import numpy as np
import pandas as pd
import os, os.path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats
import pickle
from statsmodels.distributions.empirical_distribution import ECDF
import pybobyqa
from tqdm import tqdm
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore")

def upload_static_shared(phi, alpha, delay, num_services, B, city,ppf_static,ppf_helper,ETA,upper_bound=False,lower_bound=False):
    cap_static = np.load(os.path.join(ROOT_DIR,f'{save_folder}/{city}/PHI_{phi}/ETA_{ETA}/Th_{delay}/cap_fore_uncer_test_block1_delay_{delay}_phi_{phi}_gamma_2_deltax_005.npy'))
    cap_shared=np.load(os.path.join(ROOT_DIR,f'{save_folder}/{city}/PHI_{phi}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{delay}/shared_fore_uncer_block2_delay_{delay}_phi_{phi}_alpha_{alpha}_gamma_2.npy'))
    #cap_shared = np.load(f'Results/{city}/Th_{delay}/shared_fore_uncer_block2_delay_{delay}_phi_{phi}_alpha_{alpha}_gamma_2.npy')

    dist_cd = scipy.stats.norm(loc=cap_static.mean(axis=-1),
                               scale=cap_static.std(axis=-1))
    dist_cs = scipy.stats.norm(loc=cap_shared.mean(axis=-1),
                               scale=cap_shared.std(axis=-1))
    
    ppf_static = 0.999
    ppf_helper = 0.999

    if upper_bound:
        upper_cd_static = dist_cd.ppf(ppf_static) + 2*dist_cd.std()
        upper_cs_shared = dist_cs.ppf(ppf_helper) + 2*dist_cs.std()
    elif lower_bound:
        upper_cd_static = dist_cd.ppf(ppf_static) - 2*dist_cd.std()
        upper_cs_shared = dist_cs.ppf(ppf_helper) - 2*dist_cs.std()
    else:
        upper_cd_static = dist_cd.ppf(ppf_static)
        upper_cs_shared = dist_cs.ppf(ppf_helper)
        
    
    ### Maybe this ppf values does not have to be the same as the one used in the forecast ???
    # upper_cd_static = dist_cd.ppf(ppf_static)  # Can be changed. If set to 0.5 = mean
    
    
    upper_cd_static[np.where(np.isnan(upper_cd_static))] = (
        cap_static[np.where(np.isnan(upper_cd_static))][:, 0])
    # upper_cs_shared = dist_cs.ppf(ppf_helper) # Can be changed. If set to 0.5 = mean 
    upper_cs_shared[np.where(np.isnan(upper_cs_shared))] = (
        cap_shared[np.where(np.isnan(upper_cs_shared))][:, 0])

    return upper_cd_static, upper_cs_shared

def cost_func_evaluation(c_plus, forecasting, static, phi, alpha):
    ''' Evaluate the cost of the allocation given cplus selected.'''
    # cost_shared = phi
    cost_sla = alpha
    total_cost = 0
    for i in range(forecasting.shape[0]):  # shape[0] = num services
        # total_cost += cost_shared * c_plus[i]
        ecdf = ECDF(forecasting[i])
        total_cost += (1-ecdf(static[i] + c_plus[i])) * cost_sla
    return total_cost

def apps_need_shared(forecasting, static):
    ''' Return the slices that need shared capacity '''
    index_app = []
    for app in range(forecasting.shape[0]):
        dist = ECDF(forecasting[app])
        if (1 - dist(static[app])) != 0:
            index_app.append(app)
    return index_app


def fun(pn, dist, static, max_shared, phi, alpha, lower_bound, upper_bound,obj_noise_test): # Golden search
    """ Function with only one variable to be minimized through
        bounded golden search
    """
    num_app = dist.shape[0]
    p_0 = np.ones(num_app) * 0.5
    opt = pybobyqa.solve(cost_func_evaluation_p_fix_p4, p_0,
                         bounds=(lower_bound[:num_app], upper_bound[:num_app]),
                         args=(dist, static, max_shared, pn, phi, alpha),
                         objfun_has_noise=obj_noise_test)
    return opt.f


def cost_func_evaluation_p(p, forecasting, static, shared_available, phi,
                           alpha):
    ''' Evaluate the cost of the allocation given cplus selected.'''
    # cost_shared = phi
    cost_sla = alpha
    total_cost = 0
    cplus = return_cplus(p, shared_available)
    for i in range(forecasting.shape[0]):  # shape[0] = num services
        # total_cost += cost_shared * cplus[i]
        ecdf = ECDF(forecasting[i])
        total_cost += (1-ecdf(static[i] + cplus[i])) * cost_sla
    return total_cost


def cost_func_evaluation_p_fix_p4(p, forecasting, static, shared_available, p4,
                                  phi, alpha):
    ''' Evaluate the cost of the allocation given cplus selected.'''
    # cost_shared = phi
    cost_sla = alpha
    total_cost = 0
    cplus, cplus_4 = return_cplus_fix_p4(p, shared_available, p4)
    for i in range(forecasting.shape[0]):  # shape[0] = num services
        # total_cost += cplus[i] * phi
        ecdf = ECDF(forecasting[i])
        total_cost += (1-ecdf(static[i] + cplus[i])) * cost_sla
    return total_cost


def return_cplus(p_vector, shared_available): # Transform function from the paper
    """ Return c_plus given p and max amount of shared for
    that phi,alpha and time.
    """
    cplus = np.zeros(p_vector.shape)
    products = np.zeros(p_vector.shape[0]-1)
    for i in range(len(cplus)-1):
        products[i] = np.true_divide(np.prod(p_vector[i:-1]),
                                     np.prod((1-p_vector)[i:-1]))
    cplus[-1] = np.true_divide(shared_available * p_vector[-1],
                               np.sum(products) + 1)
    for i in range(len(cplus)-1):
        cplus[i] = products[i] * cplus[-1]
    return cplus


def return_cplus_fix_p4(p_vector, shared_available, p4): # Transform function from the paper
    """ Return c_plus given p and max amount of shared for
    that phi,alpha and time. """
    cplus = np.zeros(p_vector.shape)
    products = np.zeros(p_vector.shape[0])
    for i in range(len(cplus)):
        products[i] = np.true_divide(np.prod(p_vector[i:]),
                                     np.prod((1-p_vector)[i:]))
    cplus4 = np.true_divide(shared_available * p4,
                            np.sum(products) + 1)
    # for i in range(len(cplus)):
    #     cplus[i] = products[i] * cplus4
    cplus = products * cplus4
    return cplus, cplus4


def return_p(cplus, max_shared):
    p = np.zeros(cplus.shape)
    for i in range(cplus.shape[0] - 1):
        p[i] = cplus[i] / (cplus[i]+cplus[i+1])
    p[-1] = np.sum(cplus) / max_shared
    return p



def load_real_data(city):
    ''' Load the real data for the city.'''

    bordeaux = pd.read_csv(os.path.join(ROOT_DIR,f'./citys/{city}.csv'))
    #bordeaux = pd.read_csv(f'/home/sergi_alcala/AZTEC_extension/citys/{city}.csv')

    bordeaux.drop('date_time', axis=1, inplace=True)
    bordeaux = bordeaux.reindex(sorted(bordeaux.columns), axis=1)
    bordeaux = bordeaux.to_numpy()
    return bordeaux

def load_mae_forecasting(city):
    mae_forecasting = np.transpose(np.load(os.path.join(ROOT_DIR,f'./{save_folder}/{city}/helper_forecasting_delay_1.npy')), (0,2,1))
    return mae_forecasting

def load_scalers(city, ppf_static, ppf_helper,delay,ETA):
    static_scaler = pickle.load(open(os.path.join(ROOT_DIR,f'{save_folder}/{city}/PHI_{phi}/ETA_{ETA}/Th_{delay}/block_1_minmaxscaler.pkl'), 'rb'))
    shared_scaler = pickle.load(open(os.path.join(ROOT_DIR,f'{save_folder}/{city}/PHI_{phi}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{delay}/block_2_minmaxscaler.pkl'), 'rb'))
    return static_scaler, shared_scaler

def denormalize_normalize_static_shared(static,shared, static_scaler,shared_scaler, mae_forecasting, real_data, test_index_start):
    ''' Denormalize the static data using the scaler. '''
    static_denorm = static_scaler.inverse_transform(static) # Static = block 1
    shared_denorm = shared_scaler.inverse_transform(np.expand_dims(shared, axis=-1)) # Shared = block 2
    mae_denorm = np.zeros(mae_forecasting.shape) # MAE = helper
    for i in range(B):
        mae_denorm[:,:,i] = static_scaler.inverse_transform(mae_forecasting[:,:,i])
    ''' Normalize the data to be in the same scale as the real data.'''
    shared_norm = shared_denorm / 10e9 # Shared = block 2
    static_norm = static_denorm / 10e9 # Static = block 1
    mae_norm = mae_denorm / 10e9 # MAE = helper
    output_norm = real_data[test_index_start:] / 10e9 # Real test data
    diff = min(mae_norm.shape[0], output_norm.shape[0], shared_norm.shape[0], static_norm.shape[0])

    mae_norm_diff = mae_norm.shape[0] - diff
    output_norm_diff = output_norm.shape[0] - diff
    shared_norm_diff = shared_norm.shape[0] - diff
    static_norm_diff = static_norm.shape[0] - diff

    if mae_norm_diff != 0:
        mae_norm = mae_norm[mae_norm_diff:]
    if output_norm_diff != 0:
        output_norm = output_norm[output_norm_diff:]
    if shared_norm_diff != 0:
        shared_norm = shared_norm[shared_norm_diff:]
    if static_norm_diff != 0:
        static_norm = static_norm[static_norm_diff:]

    return static_norm, shared_norm, mae_norm, output_norm

def create_df_shared_cplus_sla(slas, cplus, shared_needed,num_services,Filesave,delay,phi,alpha,upper_bound=False,lower_bound=False):
    
    shared_needed_df=pd.DataFrame(shared_needed)
    cplus_df=pd.DataFrame(cplus)
    slas_df=pd.DataFrame(slas)
    shared_cplus_sla=pd.concat([shared_needed_df, cplus_df, slas_df], axis=0)
    shared_needed_idx=[f'shared_needed_{i}' for i in range(num_services)]
    cplus_idx=[f'cplus_{i}' for i in range(num_services)]
    slas_idx=[f'slas_{i}' for i in range(num_services)]
    shared_cplus_sla.index=shared_needed_idx+cplus_idx+slas_idx
    shared_cplus_sla=shared_cplus_sla.T
    shared_needed_df=shared_needed_df.T
    cplus_df=cplus_df.T
    slas_df=slas_df.T

    if upper_bound:
        shared_cplus_sla.to_csv(f'{Filesave}/Block_3_shared_cplus_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_UPPER_BOUND.csv',index=False)
        shared_needed_df.to_csv(f'{Filesave}/Block_3_shared_needed_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_UPPER_BOUND.csv',index=False)
        cplus_df.to_csv(f'{Filesave}/Block_3_cplus_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_UPPER_BOUND.csv',index=False)
        slas_df.to_csv(f'{Filesave}/Block_3_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_UPPER_BOUND.csv',index=False)
    elif lower_bound:
        shared_cplus_sla.to_csv(f'{Filesave}/Block_3_shared_cplus_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_LOWER_BOUND.csv',index=False)
        shared_needed_df.to_csv(f'{Filesave}/Block_3_shared_needed_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_LOWER_BOUND.csv',index=False)
        cplus_df.to_csv(f'{Filesave}/Block_3_cplus_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_LOWER_BOUND.csv',index=False)
        slas_df.to_csv(f'{Filesave}/Block_3_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_LOWER_BOUND.csv',index=False)
    else:
        shared_cplus_sla.to_csv(f'{Filesave}/Block_3_shared_cplus_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv',index=False)
        shared_needed_df.to_csv(f'{Filesave}/Block_3_shared_needed_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv',index=False)
        cplus_df.to_csv(f'{Filesave}/Block_3_cplus_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv',index=False)
        slas_df.to_csv(f'{Filesave}/Block_3_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv',index=False)


def allocation_loop(static_norm, shared_norm, mae_norm, output_norm, num_services,  phi, alpha,delay, Filesave,upper_boundari=False,lower_boundari=False):

    length_tsteps = min(len(shared_norm),len(output_norm))
    slas = np.zeros((num_services, length_tsteps)) # number of slas for each time step
    cplus = np.zeros((num_services, length_tsteps)) # shared allocated to each app for each time step
    shared_needed=np.zeros((num_services, length_tsteps)) # shared needed for each app for each time step ### Sergi ###
    p = np.zeros((num_services, length_tsteps)) 
    lower_bound = np.ones(num_services) * 1e-10
    upper_bound = np.ones(num_services) * 0.99999
    idx_time = 1
    try:
        for time in tqdm(range(0, length_tsteps)):
            forecasting = mae_norm[time] # forecasting.shape = (5, 100)
            idx_app = apps_need_shared(forecasting, static_norm[time])
            if len(idx_app) > 1:
            ##Try catch used because the assymetry  of the matrix, solved by changing the parameter objfun_has_noise True or False
            
                try:
                    obj_noise_test = True
                # Golden search for setting p4
                    test = scipy.optimize.minimize_scalar(fun, bounds=(1e-10, 0.99999), method='bounded',
                                                        args=(forecasting[idx_app[:-1]], static_norm[time, idx_app[:-1]],
                                                                shared_norm[time], phi, alpha, lower_bound, upper_bound,obj_noise_test))
                except:
                    print (f'obj_noise_test = False for time {time}')
                    obj_noise_test = False
                    test = scipy.optimize.minimize_scalar(fun, bounds=(1e-10, 0.99999), method='bounded',
                                                        args=(forecasting[idx_app[:-1]], static_norm[time, idx_app[:-1]],
                                                                shared_norm[time], phi, alpha, lower_bound, upper_bound,obj_noise_test))
                p4 = test.x # p4 obtained using golden search
                p_0 = np.ones(len(idx_app)-1) * 0.5 # initial values of p from 0 to 3
            # Finding the best initial values of p from 0 to 3 keeping p4 fixed. BOBYQA for p from 0 to 3.
                try:
                    
                    prova = pybobyqa.solve(cost_func_evaluation_p_fix_p4, p_0, bounds=(lower_bound[:len(idx_app)-1],
                                                                                    upper_bound[:len(idx_app)-1]),
                                        args=(forecasting[idx_app[:-1]], static_norm[time, idx_app[:-1]],
                                                shared_norm[time], p4, phi, alpha), objfun_has_noise=True)
                    #print('objfun_has_noise=True for prova')
                except:
                    print (f'objfun_has_noise=False for time {time} in prova')
                    prova = pybobyqa.solve(cost_func_evaluation_p_fix_p4, p_0, bounds=(lower_bound[:len(idx_app)-1],
                                                                                upper_bound[:len(idx_app)-1]),
                                    args=(forecasting[idx_app[:-1]], static_norm[time, idx_app[:-1]],
                                            shared_norm[time], p4, phi, alpha), objfun_has_noise=False)
                    #print('objfun_has_noise=False for prova')
                p_0 = np.zeros(len(idx_app))
                p_0[:len(idx_app)-1] = prova.x
                p_0[-1] = p4
            # Now we have the "best" initial values for p, so we run the allocation algorithm for all p. BOBYQA for all p.
                try:
                    
                    prova_2 = pybobyqa.solve(cost_func_evaluation_p, p_0, bounds=(lower_bound[:len(idx_app)],
                                                                                upper_bound[:len(idx_app)]),
                                            args=(forecasting[idx_app], static_norm[time, idx_app],
                                                shared_norm[time], phi, alpha), objfun_has_noise=True)
                    #print('objfun_has_noise=True for prova_2')
                except:
                    # print (f'objfun_has_noise=False for time {time} in prova_2')
                    prova_2 = pybobyqa.solve(cost_func_evaluation_p, p_0, bounds=(lower_bound[:len(idx_app)],
                                                                            upper_bound[:len(idx_app)]),
                                        args=(forecasting[idx_app], static_norm[time, idx_app],
                                            shared_norm[time], phi, alpha), objfun_has_noise=False)
                #print('objfun_has_noise=False for prova_2')
                cplus[idx_app, time] = return_cplus(prova_2.x, shared_norm[time]) # cplus = shared capacity allocated to each service.
                p[idx_app, time] = prova_2.x
            elif len(idx_app) == 1:
            # In case we have only one application requesting shared capacity, I assign all the shared available to it.
                cplus[idx_app, time] = shared_norm[time]
            shared_needed[:, time] = output_norm[time] - static_norm[time] # It gives you the total shared required at that time.
            for app in range(num_services):
                if shared_needed[app, time] > cplus[app, time]:
                    slas[app, time] += 1
            #print(f'\rTime={time}, {idx_app}, Shared needed: {shared_needed[:,time]}, slas: {slas[:, time]}, cplus: {cplus[:, time]}', end='')
        create_df_shared_cplus_sla(slas, cplus, shared_needed,num_services,Filesave,delay,phi,alpha,upper_bound=upper_boundari,lower_bound=lower_boundari) ###Sergi
    except:
        with open ('/home/sergi_alcala/sergi_data/AZTEC_extension/Results/Error.txt', 'a') as f:
            error_txt=f'Error in city {city}, phi {phi}, alpha {alpha}, ppf_static {ppf_static}, ppf_helper {ppf_helper}, delay {delay}, ETA {ETA}'
            print(error_txt)
            f.write(error_txt+'\n')
           
       
        pass

def fun_test_index_start(city,delay):
    bordeaux= load_real_data(city)
    test_index_start = round(len(bordeaux)*0.8) + delay
    return test_index_start




def myMultiOpt(pair):
    city,phi,alpha,ppf_helper,ppf_static,ETA,bound=pair
    fpath=f'{ROOT_DIR}{save_folder}/{city}/PHI_{phi}/ETA_{ETA}'
    delay=np.load(f'{fpath}/overall_optimal_window.npy')
    # delay= 5
    print('delay',delay)    

    if bound == 'upper':
        upper_bound=True
        lower_bound=False
    elif bound == 'lower':
        upper_bound=False
        lower_bound=True
    else:
        upper_bound=False
        lower_bound=False

    Filesave=f'{fpath}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{delay}'
    print(Filesave)
    print(os.path.join(Filesave,f'/Block_3_shared_cplus_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv'))
    Filexists = os.path.join(Filesave,f'/Block_3_shared_cplus_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv')
                    
    # if not os.path.exists(os.path.join(Filesave,f'/Block_3_shared_cplus_sla_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv')):
    # if not os.path.exists(Filexists):
    if upper_bound:
        path=f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{phi}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{delay}/Block_3_cplus_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_UPPER_BOUND.csv'
    elif lower_bound:
        path=f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{phi}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{delay}/Block_3_cplus_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2_LOWER_BOUND.csv'
    else:
        path=f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{phi}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{delay}/Block_3_cplus_delay_{delay}_phi_{phi}_alpha{alpha}_gamma_2.csv'

    if not os.path.exists(path):

        print(path)

        
    
        test_index_start=fun_test_index_start(city,delay)
        print(f'City: {city}, ppf_static: {ppf_static}, ppf_helper: {ppf_helper}, phi: {phi}, alpha: {alpha}, ETA: {ETA}')
        real_data, mae_forecasting=load_real_data(city),load_mae_forecasting(city)
        static_scaler, shared_scaler = load_scalers(city, ppf_static, ppf_helper,delay,ETA)
        static,shared=upload_static_shared(phi, alpha, delay, num_services, B, city,ppf_static,ppf_helper,ETA,upper_bound=upper_bound,lower_bound=lower_bound)
        # print(ETA)
        static_norm, shared_norm, mae_norm, output_norm = denormalize_normalize_static_shared(static, shared, static_scaler, shared_scaler, mae_forecasting, real_data, test_index_start)
        allocation_loop(static_norm, shared_norm, mae_norm, output_norm, num_services,  phi, alpha,delay,Filesave,upper_boundari=upper_bound,lower_boundari=lower_bound)
    else:
       print(f'Already done: City: {city}, ppf_static: {ppf_static}, ppf_helper: {ppf_helper}, phi: {phi}, alpha: {alpha}')



# ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))
# print(f'Root dir: {ROOT_DIR}')
ROOT_DIR = '/home/sergi_alcala/sergi_data/AZTEC_extension/'
lookback = 6
num_services = 5

B = 100 # Number of montecarlo output

# 
PHIS = [10] # Negative slope of the loss function TO BE MODIFIED 

# cities = ['Dijon', 'Grenoble', 'Lille', 'Lyon', 'Marseille', 'Montpellier', 'Nantes',
#         'Nice', 'Paris', 'Reims', 'Rennes', 'Strasbourg']
cities = ['Paris']

ALPHAS=[2]
# ALPHAS=[2]




# TB_Fpath = 'Training_Block_Optimal_Cities'
# save_folder = 'Results_Optimal_Cities_denorm'



# TB_Fpath = 'Training_Block_Optimal_Cities_loss_RAW'
# save_folder = 'Results_Optimal_Cities_denorm_loss_RAW'


# TB_Fpath = 'Training_Block_Optimal_Cities_test_kr_ki'
# save_folder = 'Results_Optimal_Cities_test_kr_ki'
# save_folder = 'Results_AZTEC_6'
save_folder = 'Results_Optimal_Cities_test_kr_ki_ALLOC_changed'


pair_list=[]
# ppf_static_list=[0.9,0.99]
ppf_static_list=[0.9]
ppf_helper_list=[0.7]
ETAS = [1,2,10,20,30,40,50,70,90,100]
# ETAS = [1]

# bound = 'upper'
bound = 'lower'
# bound = 'none'


pair_list=[]
for city in cities:
    for ppf_helper in ppf_helper_list:
        for ppf_static in ppf_static_list:
            for phi in PHIS:
                for alpha in ALPHAS:
                    for ETA in ETAS:
                        pair_list.append((city,phi,alpha,ppf_helper,ppf_static,ETA,bound))

if __name__ == '__main__':
    with Pool(10) as p:
        p.map(myMultiOpt,pair_list)

                
                    
