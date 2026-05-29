import sys, os

sys.path.append('../')
from Libraries import *
from Diagram_Call import *

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import multiprocessing


multiprocessing.set_start_method('spawn', force=True)
       
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)






def myMultiOpt(pair):
    
    city,ETA,PHI,ALPHA,SEL_SERV_H,n_simulation,Simulations,ppf_helper_list,ppf_static_list = pair
    n_simulation=int(n_simulation)


    config=dl.get_config(city,ETA, LOOKBACK, DELAY_Block1_Block2,DELAY_Helper, PHI, GAMMA, NUM_SERV_B1,NUM_SERV_H,SEL_SERV_H,
    EPOCHS_block1,EPOCHS_block2,EPOCHS_Helper, B,ALPHA,n_simulation,ppf_helper=ppf_helper_list,ppf_static=ppf_static_list)

    
    
    DB1T_scipy.Diagram_Block_1_Training_scipy(config,TB_Fpath,save) ### Scipy version for obtaining the optimal window






    
    # config = DOWS.Diagram_Optimal_Window_Selection(config,PHI,TB_Fpath,save_folder,Simulations) # we update the optimal window after all simulations

    # config['DELAY_Block1_Block2']=int(np.load(f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/overall_optimal_window.npy'))
    # config['DELAY_Block1_Block2'] = 5
    

    # DB1.Diagram_Block_1(config,save_folder)

    # DH.Diagram_Helper(config,save_folder)

 

    # DB2.Diagram_Block_2(config,save_folder)




LOOKBACK = 6 # History given as input to the network. Could be modified if needed
GAMMA = 2 # Positive slope of the loss function
NUM_SERV_B1 = 5 # number of services for the block 1
NUM_SERV_H = 1 # number of services for the helper
B = 100 # Number of montecarlo output
SEL_SERVS=[0, 1, 2, 3, 4] #Services to be selected for Helper Block


PHIS = [0.1,0.5,1,10] # Negative slope of the loss function TO BE MODIFIED 
# PHIS = [10]

# cities = ['Dijon', 'Grenoble', 'Lille', 'Lyon', 'Marseille', 'Montpellier', 'Nantes',
#         'Nice', 'Paris','Reims', 'Rennes', 'Strasbourg'] ## Bordeaux and Toulouse are not included in the dataset 

# cities = ['Dijon']
cities = ['Paris']
# ALPHAS=[2,3,5]
ALPHAS=[2]

save=False

DELAY_Block1_Block2 = None # Lenght forecasted by the network. Could be modified if needed
DELAY_Helper = 1 # Lenght forecasted by the network. Could be modified if needed
EPOCHS_block1 = 300 # Number of epochs for the block 1
EPOCHS_block2 = 500 # Number of epochs for the block 2
EPOCHS_Helper = 300 # Number of epochs for the helper
Simulations= 10 # Number of simulations for the optimal window selection


TB_Fpath = 'TRAINING_FLOPS'
save_folder = 'Results_Optimal_Cities_test_kr_ki_new'
# save_folder = 'Results_Optimal_Cities_test_kr_ki_ALLOC_changed_b2'

pair_list=[]
# ppf_static_list=[0.9,0.99]
ppf_static_list=[0.9]
ppf_helper_list=[0.7]

ETAS =[1,2,10,20,30,40,50,70,90,100]
# ETAS = [2]
# ETAS =[20]

for city in cities:
    for PHI in PHIS:
        for ETA in ETAS:
            for ALPHA in ALPHAS:
                for ppf_helper in ppf_helper_list:
                    for ppf_static in ppf_static_list:
                        for n_simulation in range(Simulations):
                        
                        
                            pair_list.append((city,ETA,PHI,ALPHA,SEL_SERVS,n_simulation,Simulations,ppf_helper,ppf_static))
        

def get_flops_block_1(config):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            # Build and compile the Block 1 model
            model = DB1T_scipy.get_model_block_1(config)
            model.compile(optimizer='adam', loss='mse')

            # Profile the graph to calculate FLOPS
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='scope', options=opts)

    return flops.total_float_ops / 1e12 if flops else 0  # Convert to TFLOPS


def get_flops_block_2(config):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            # Build and compile the Block 2 model
            model = DB2.get_model_block_2(config)
            model.compile(optimizer='adam', loss='mse')

            # Profile the graph to calculate FLOPS
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='scope', options=opts)

    return flops.total_float_ops / 1e12 if flops else 0  # Convert to TFLOPS


if __name__ == '__main__':
    with Pool(45) as p:
        p.map(myMultiOpt,pair_list)

    for pair in pair_list:
        config = dl.get_config(*pair[:9], ppf_helper=pair[7], ppf_static=pair[8])
        print(f"Block 1 TFLOPS: {get_flops_block_1(config)}")
        print(f"Block 2 TFLOPS: {get_flops_block_2(config)}")

