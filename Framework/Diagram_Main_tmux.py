from Libraries import *
from Diagram_Call import *

import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
       
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


LOOKBACK = 6 # History given as input to the network. Could be modified if needed

GAMMA = 2 # Positive slope of the loss function
NUM_SERV_B1 = 5 # number of services for the block 1
NUM_SERV_H = 1 # number of services for the helper

#B = 100 # Number of montecarlo output
B=100
#ppf_helper = 0.75 # Percentile of the helper
#ppf_static = 0.1 # Percentile of the static model



SEL_SERVS=[0, 1, 2, 3, 4] #Services to be selected for Helper Block
# PHIS = [0.1,1,10] # Negative slope of the loss function TO BE MODIFIED 
PHIS = [0.1] # Negative slope of the loss function TO BE MODIFIED 
# cities = ['Bordeaux','Dijon', 'Grenoble', 'Lille', 'Lyon', 'Marseille', 'Montpellier', 'Nantes',
#         'Nice', 'Paris', 'Reims', 'Rennes', 'Strasbourg', 'Toulouse']
cities = ['Paris']

ALPHAS=[2,3,5]
#ALPHAS=[2]
save=True

DELAY_Block1_Block2 = None # Lenght forecasted by the network. Could be modified if needed
DELAY_Helper = 1 # Lenght forecasted by the network. Could be modified if needed
EPOCHS_block1 = 100 # Number of epochs for the block 1
EPOCHS_block2 = 500 # Number of epochs for the block 2
EPOCHS_Helper = 150 # Number of epochs for the helper
# EPOCHS_block1 = 1 # Number of epochs for the block 1
# EPOCHS_block2 = 1 # Number of epochs for the block 2
# EPOCHS_Helper = 1 # Number of epochs for the helper
Simulations= 10 # Number of simulations for optimal window selection


# TB_Fpath=f'Training_block_scipy_099'
# TB_Fpath=f'/home/sergi_alcala/sergi_data/AZTEC_extension/Training_block_bounds_2_100_sim_100_b_1'
# TB_Fpath='TEST_4_H_new'
TB_Fpath = 'Training_Block_Optimal_Cities_test_kr_ki'
pair_list=[]

save=True


# save_folder='Test_lengths'
# save_folder='Results_100_sim_bounds_2_100'
# save_folder='results_TEST_4_H_new_helper'

save_folder = 'Results_Optimal_Cities_test_kr_ki'
ETAS=[1,2,10,20]
n_simulation=0


####Training block (AZTEC extension)
for city in cities:
        # config=dl.get_config(city, LOOKBACK, DELAY_Block1_Block2,DELAY_Helper, PHI, GAMMA, NUM_SERV_B1,NUM_SERV_H,SEL_SERVS,
                        # EPOCHS_block1,EPOCHS_block2,EPOCHS_Helper, B,ALPHAS,n_simulation)
    
#     DH.Diagram_Helper(config,save_folder,save) # We only need to run one Helper per city, it does not depend of alpha, phi, etc.
    
        for PHI in PHIS:
                for eta in ETAS:
        
        #     for n_simulation in range(Simulations):
        
            
                        config=dl.get_config(city,eta, LOOKBACK, DELAY_Block1_Block2,DELAY_Helper, PHI, GAMMA, NUM_SERV_B1,NUM_SERV_H,SEL_SERVS,
                        EPOCHS_block1,EPOCHS_block2,EPOCHS_Helper, B,ALPHAS[0],n_simulation)
                
  
                DB1T_scipy.Diagram_Block_1_Training_scipy(config,TB_Fpath,save) ### Scipy version for obtaining the optimal window

        # config=dl.get_config(city, LOOKBACK, DELAY_Block1_Block2,DELAY_Helper, PHI, GAMMA, NUM_SERV_B1,NUM_SERV_H,SEL_SERVS,
        #                 EPOCHS_block1,EPOCHS_block2,EPOCHS_Helper, B,ALPHAS,n_simulation)

                        # config = DOWS.Diagram_Optimal_Window_Selection(config,PHI,TB_Fpath,save_folder,Simulations) # we update the optimal window after all simulations
        # config['DELAY_Block1_Block2']=int(np.load(f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/overall_optimal_window.npy'))
        
        
        # DB1.Diagram_Block_1(config,save_folder,save) # We then run block 1 with the optimal window for all phis, it does not depend of alpha
       
        # for ALPHA in ALPHAS:
        #         # config['DELAY_Block1_Block2']=83
        #         config['ALPHA']=ALPHA 
        #         DB2.Diagram_Block_2(config,save_folder,save) # We then run block 2 with the optimal window for all phis and alphas



               

                
               




# config=dl.get_config(city,XTRAIN, LOOKBACK, DELAY_Block1_Block2,DELAY_Helper, PHI, GAMMA, NUM_SERV_B1,NUM_SERV_H,SEL_SERV_H,
#  EPOCHS_block1,EPOCHS_block2,EPOCHS_Helper, B,ALPHA,ppf_helper,ppf_static)

# if not os.path.exists(f'./AZTEC_extension/Results/{city}/Th_{DELAY_Block1_Block2}'):
#     os.makedirs(f'./AZTEC_extension/Results/{city}/Th_{DELAY_Block1_Block2}')












