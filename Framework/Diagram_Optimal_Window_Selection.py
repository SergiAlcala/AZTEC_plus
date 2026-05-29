try:
    from Libraries import *
except:

    from Framework.Libraries import *


def parse_val_loss(config, fpath,PHI,Simulations,ETA=False):    
    optimall_windo_list=[]
    optimal_all = []
    opttimal_val_loss=[]



    
    city=config['city']
    PHI=config['PHI']

    for Simulation in range( Simulations):

        if ETA :
            Fpath=f'{fpath}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/optimal_window.npy'

        else:
        
            Fpath=f'{fpath}/PHI_{PHI}/Simulation_{Simulation}/optimal_window.npy'
        optimal_window=np.load(Fpath)
        optimal_all.append(optimal_window)

        optimal_int=int(optimal_window)
        if ETA :
            pred_fpath=f'{fpath}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/Th_{optimal_int}/training_block1_delay_{optimal_int}_phi_{PHI}_gamma_2_deltax_005.npy'
            
        else:
            pred_fpath=f'{fpath}/PHI_{PHI}/Simulation_{Simulation}/Th_{optimal_int}/training_block1_delay_{optimal_int}_phi_{PHI}_gamma_2_deltax_005.npy'
        pred=np.load(pred_fpath)
        pred_montecarlo=montecarlo_stuff(pred)
        x_val_norm_towindow,minmaxscaler=load_city_val_norm(config)

        val_loss= denorm_validation_loss(pred_montecarlo, x_val_norm_towindow, minmaxscaler,PHI,ETA)

        optimall_windo_list.append(int(optimal_window))
        opttimal_val_loss.append(val_loss)
        
       

    return optimall_windo_list,opttimal_val_loss

def select_optimal_window(optimall_windo_list,opttimal_val_loss):

    df=pd.DataFrame(opttimal_val_loss).T
    df.columns=optimall_windo_list
    opt_window=int(df.idxmin(axis=1))
    print(f'Optimal window is {opt_window}')
    # print(df)
    return opt_window

def montecarlo_stuff(static_val,ppf_val=0.7):
    static_val = static_val.clip(min=0)
                    
            # # Save results of the test model with validation dataset 
            


    dist_load_forecasted = scipy.stats.norm(loc=static_val.mean(axis=-1),
                                        scale=static_val.std(axis=-1))


    upper_load_forecasted = dist_load_forecasted.ppf(ppf_val) 

    upper_load_forecasted[np.where(np.isnan(upper_load_forecasted))] = (
    static_val[np.where(np.isnan(upper_load_forecasted))][:, 0])
    return upper_load_forecasted


def Diagram_Optimal_Window_Selection(config,PHI,T_Block_Folder,save_folder,Simulations):

    ## Set Variables
    city = config['city']
    GAMMA = config['GAMMA']
    PHI = config['PHI']
    ETA = config['ETA']
     


    fpath=f'/home/sergi_alcala/sergi_data/AZTEC_extension/{T_Block_Folder}/{city}'

    optimall_windo_list,opttimal_val_loss = parse_val_loss(config,fpath,PHI,Simulations,ETA)
    optimal_window = select_optimal_window(optimall_windo_list,opttimal_val_loss)
    print(f'Optimal window for {city} is {optimal_window} with PHI={PHI} and ETA={ETA}')
    config['DELAY_Block1_Block2']=optimal_window
    config['PHI']=PHI
    if not os.path.exists(f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}'):
        os.makedirs(f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}')

    np.save(f'/home/sergi_alcala/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/overall_optimal_window.npy',optimal_window)
    
    return config



def define_delta_i(pred,real,num_services):
    delta_i = np.zeros((pred.shape[0], num_services)) 


    indexes_under = np.where((pred - real) > 0)
    indexes_above = np.where((pred - real) < 0)

    # Where the real load is below the allocated static, delta_i is equal to the real load.
    # In this case rho_i is equal to 0.

    for idx, element in enumerate(indexes_under[0]):
        delta_i[element, indexes_under[1][idx]] = real[element, indexes_under[1][idx]]


    # Where the real load is higher than the allocated static, delta_i is equal to the static denormalized.
    # In this case rho_i is equal to the difference between real load and static allocated.
        
    for idx, element in enumerate(indexes_above[0]):
        delta_i[element, indexes_above[1][idx]] = pred[element, indexes_above[1][idx]]

    return delta_i
def instantiation_cost_calculation(pred,val_length,num_services,eta,delta_i):

    
    delta_static = pred[:val_length] - np.roll(pred[:val_length], 1,axis=0)
    delta_static[0] = pred[0]

    cost_instantiation_static = np.zeros((val_length, num_services))
    cost_instantiation_static[np.where(delta_static > 0)] = eta * delta_i[np.where(delta_static > 0)]

    cost_instantiation_static = cost_instantiation_static.mean()
    return cost_instantiation_static


# Definition of function to return the denormalized validation loss
def denorm_validation_loss(pred, act, scaler,PHI,ETA):

    diff_shape = abs(pred.shape[0] - act.shape[0])

    if len(pred)> len(act):
        pred=pred[diff_shape:]
    elif len(pred)< len(act):
        act=act[diff_shape:]

   

    # Denormalize predicted and real load and get denormalized peak
    actuals=scaler.inverse_transform(act)
    preds=scaler.inverse_transform(pred)
    
    
    # Get the denormalized loss from overprovisioning and SLA violations
    loss= evaluate_costs_single_clust(preds, actuals,PHI)

    
    delta_i=define_delta_i(preds,actuals,actuals.shape[1])

    eta = ETA # cost of instantiation

    intantiation_cost= instantiation_cost_calculation(preds,preds.shape[0],preds.shape[1],eta,delta_i)

    total_val_loss = loss+intantiation_cost
    
    return total_val_loss
    
  

GAMMA = 2
def cost_func(y_true, y_pred, reduction='none'):
    slope = PHI
    gamma = GAMMA
    delta_x = 0.05 # Can be removed or modified
    forecast_delay = DELAY
    y_pred = tf.expand_dims(y_pred, axis=1)
    y_pred = tf.tile(y_pred, (1, forecast_delay, 1))
    cost = y_pred - y_true - delta_x
    pen_positive = gamma * cost
    pen_negative = slope * cost
    cost = tf.where(cost > 0, pen_positive, cost)
    cost = tf.where(cost <= 0, pen_negative, cost)
    cost = tf.abs(cost)
    cost = K.sum(K.sum(cost, axis=-1), axis=-1) # In this way, we have a value for each element of the batch otherwise you can also average over the batch
    return cost
    

def cost_func_window(y_true, y_pred,PHI, reduction='none'):
    slope = PHI
    gamma = GAMMA
    delta_x = 0.05 # Can be removed or modified
    
    cost = y_pred - y_true - delta_x
    pen_positive = gamma * cost
    pen_negative = slope * cost
    cost = np.where(cost > 0, pen_positive, cost)
    cost = np.where(cost <= 0, pen_negative, cost)
    cost = np.abs(cost)
    cost = np.sum(cost) # In this way, we have a value for each element of the batch otherwise you can also average over the batch
    cost = np.mean(cost)

    return cost
    


def evaluate_costs_single_clust(pred_load, real_load,PHI):
    #
    
    loss= cost_func_window( real_load,pred_load,PHI )
  
    return loss


def load_city_val_norm(config):

    city=config['city']

    bordeaux = pd.read_csv(f'/home/sergi_alcala/sergi_data/AZTEC_extension/citys/{city}.csv')
    bordeaux.drop('date_time', axis=1, inplace=True)
    bordeaux = bordeaux.reindex(sorted(bordeaux.columns), axis=1)
    bordeaux = bordeaux.to_numpy()

    xTrain_size=round(len(bordeaux)*0.7) ### maybe 0.8
    xVal_size=round(len(bordeaux)*0.1)
    xTest_size=round(len(bordeaux)*0.2)

    ## Normalize Data
    minmaxscaler = MinMaxScaler()
    x_train = bordeaux[:xTrain_size]
    x_train_norm=minmaxscaler.fit_transform(x_train)

    x_val = bordeaux[xTrain_size:xTrain_size+xVal_size]
    x_val_norm_towindow=minmaxscaler.transform(x_val)

    return x_val_norm_towindow,minmaxscaler
                    
