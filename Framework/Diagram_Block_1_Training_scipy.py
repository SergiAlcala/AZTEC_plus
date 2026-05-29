from Libraries import *
from sklearn.metrics import mean_squared_error
from Diagram_Call import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: All messages are logged (default), 1: INFO messages are not displayed, 2: INFO and WARNING messages are not displayed, 3: INFO, WARNING, and ERROR messages are not displayed

batch_size=128

def save_numpy_array_with_unique_name(base_filename, array):
    """
    Saves a NumPy array to a file. If the file already exists, adds a number to the filename.

    Parameters:
    - base_filename: str, the base filename to save the array to, without extension.
    - array: np.array, the NumPy array to save.
    """
    # Initialize variables
    counter = 0
    filename = base_filename + ".npy"

    # Loop to find a unique filename
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}_{counter}.npy"

    # Save the array once a unique filename is determined
    np.save(filename, array)
    print(f"Array saved to {filename}")

def instantiation_cost_calculation(pred,val_length,num_services,eta,delta_i):

    
    delta_static = pred[:val_length] - np.roll(pred[:val_length], 1,axis=0)
    delta_static[0] = pred[0]

    cost_instantiation_static = np.zeros((val_length, num_services))
    cost_instantiation_static[np.where(delta_static > 0)] = eta * delta_i[np.where(delta_static > 0)]

    cost_instantiation_static = cost_instantiation_static.mean(axis = 0)
    cost_instantiation_static = cost_instantiation_static.sum()

    return cost_instantiation_static


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

def define_delta_i_cf(pred,real,num_services):
    
    batch_size=pred.shape[0]
    delay = pred.shape[1]
    delta_i = np.zeros((pred.shape[0]*pred.shape[1], num_services)) 
    
    pred=pred.reshape(-1,num_services)
    real=real.reshape(-1,num_services)
 
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
    
    delta_i=delta_i.reshape(batch_size,delay,num_services)
    return delta_i

def instantiation_cost_calculation_cf(pred,val_length,delay,num_services,eta,delta_i):

    
    delta_static = pred[:val_length] - np.roll(pred[:val_length], 1,axis=0)
    delta_static[0] = pred[0]

    delta_static= delta_static.reshape(-1,num_services)
    delta_i=delta_i.reshape(-1,num_services)

    cost_instantiation_static = np.zeros((delta_static.shape[0], num_services))
    cost_instantiation_static[np.where(delta_static > 0)] = eta * delta_i[np.where(delta_static > 0)]

    cost_instantiation_static = cost_instantiation_static.reshape(val_length,delay,num_services)

    
    return cost_instantiation_static


# Definition of function to return the denormalized validation loss
def denorm_validation_loss(pred, act, scaler,DELAY):

    preds=pred
    actuals=act

    len_diff = abs(len(preds) - len(actuals))
    if len(preds)> len(actuals):
        preds = preds[len_diff:]

        
    elif len(preds)< len(actuals):
        actuals = actuals[len_diff:]

    
    # Get the denormalized loss from overprovisioning and SLA violations
    loss= evaluate_costs_single_clust(preds, actuals) 
    if not os.path.exists(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/val_loss/'):
        os.makedirs(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/val_loss/')

    save_numpy_array_with_unique_name(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/val_loss/val_loss_delay_{DELAY}', loss)
   
    

    delta_i=define_delta_i(preds,actuals,actuals.shape[1])

    eta = ETA # cost of instantiation

    intantiation_cost= instantiation_cost_calculation(preds,preds.shape[0],preds.shape[1],eta,delta_i)
    
    save_numpy_array_with_unique_name(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/val_loss/val_loss_instantation_{DELAY}', intantiation_cost)

    

    total_val_loss = loss+intantiation_cost

    save_numpy_array_with_unique_name(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/val_loss/val_loss_total_{DELAY}', total_val_loss)

    
    
    return total_val_loss

def cost_func(y_true, y_pred,DELAY, reduction='none'):
    slope = PHI
    # alpha = 2
    gamma = GAMMA
    forecast_delay = DELAY

    ################################# Block 1 Cost function ##################################
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
    
    
    # print(f'cost_mean:{cost_mean.numpy()}')
    # cost = K.sum(K.sum(cost, axis=-1), axis=-1) # In this way, we have a value for each element of the batch otherwise you can also average over the batch
    cost= K.mean(cost)
    cost_np =cost.numpy()
    ###########################################################################

    ############################## COST OF INSTANTIATION #######################

    # y_pred = y_pred.numpy()
    # y_true = y_true.numpy()
    # delta_i=define_delta_i_cf(y_pred,y_true,y_true.shape[-1])


    # eta = ETA # cost of instantiation parameter

    # intantiation_cost= instantiation_cost_calculation_cf(y_pred,y_pred.shape[0],DELAY,y_pred.shape[-1],eta,delta_i)
    # intantiation_cost = tf.convert_to_tensor(intantiation_cost, dtype=tf.float32)

    # intantiation_cost = K.sum(K.sum(intantiation_cost, axis=-1), axis=-1)
    
    # intantiation_cost= K.mean(intantiation_cost)

    # ###########################################################################
    # total_val_loss=cost+intantiation_cost
    total_val_loss=cost
    return total_val_loss
    

def cost_func_window(y_true, y_pred, reduction='none'):
    slope = PHI
    gamma = GAMMA
    delta_x = 0.05 # Can be removed or modified
 
    cost = y_pred - y_true - delta_x
    pen_positive = gamma * cost
    pen_negative = slope * cost
    cost = np.where(cost > 0, pen_positive, cost)
    cost = np.where(cost <= 0, pen_negative, cost)
    cost = np.abs(cost)
    # cost = np.sum(cost) # In this way, we have a value for each element of the batch otherwise you can also average over the batch
    cost = np.mean(cost,axis = 0)
    cost = np.sum(cost)
   
    return cost
    


def evaluate_costs_single_clust(pred_load, real_load):

    loss = cost_func_window( real_load,pred_load )

    return loss


def montecarlo_stuff(static_val,ppf_val=0.7):
    
    static_val = static_val.clip(min=0)
                    
    dist_load_forecasted = scipy.stats.norm(loc=static_val.mean(axis=-1),
                                        scale=static_val.std(axis=-1))
    
    upper_load_forecasted = dist_load_forecasted.ppf(ppf_val) 

    upper_load_forecasted[np.where(np.isnan(upper_load_forecasted))] = (
    static_val[np.where(np.isnan(upper_load_forecasted))][:, 0])

    return upper_load_forecasted



def Training_Block(DELAY):
    
    ## Load Data

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
    x_train_norm = tf.convert_to_tensor(minmaxscaler.fit_transform(x_train), dtype=tf.float32)
    x_val = bordeaux[xTrain_size:xTrain_size+xVal_size]
    x_val_norm_towindow=minmaxscaler.transform(x_val)
    x_val_norm = tf.convert_to_tensor(minmaxscaler.transform(x_val), dtype=tf.float32)
    x_test = bordeaux[xTrain_size+xVal_size:]
    x_test_norm = tf.convert_to_tensor(minmaxscaler.transform(x_test), dtype=tf.float32)
    # x_test = bordeaux[XTRAIN:]
    # x_test_norm = tf.convert_to_tensor(minmaxscaler.transform(x_test), dtype=tf.float32)

    DELAY=int(DELAY)
  

    
    print (f'City : {city}, ETA: {ETA}, PHI: {PHI}, DELAY: {DELAY}, Simulation: {Simulation}')
    
    input_dataset = keras.preprocessing.timeseries_dataset_from_array(x_train_norm[:-DELAY], None,
                                                            sequence_length=LOOKBACK, sequence_stride=DELAY)
    target_dataset = keras.preprocessing.timeseries_dataset_from_array(x_train_norm[LOOKBACK:], None,
                                                                        sequence_length=DELAY, sequence_stride=DELAY)


    val_dataset = keras.preprocessing.timeseries_dataset_from_array(x_val_norm[:-DELAY], None,
                                                                        sequence_length=LOOKBACK, sequence_stride=DELAY,
                                                                        batch_size=128)
    # test_dataset = keras.preprocessing.timeseries_dataset_from_array(x_test_norm[:-DELAY], None,
    #                                                                     sequence_length=LOOKBACK, sequence_stride=DELAY,
    #                                                                     batch_size=128)
                
    model =NN.make_model_lstm(x_train_norm.shape, LOOKBACK, NUM_SERV)
  
    optimizer = Adam(learning_rate=0.001)

    

    ## Train Model
    loss_value_list=[]
    for epoch in tqdm(range(EPOCHS)):
  
        for step, (x_batch_train, y_batch_train) in enumerate(zip(input_dataset, target_dataset)):
            with tf.GradientTape() as tape:
                prediction = model(x_batch_train, training=True)
                loss_value = cost_func(y_batch_train, prediction,DELAY)
                # print(f'loss_value:{loss_value.numpy()}')
               

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        loss_value_list.append(loss_value.numpy())

    load_forecasted = np.zeros((int(x_val_norm.shape[0]/DELAY), 5, B)) # 290 = lenght test dataset divided by delay
    actuals = np.zeros((int(x_val_norm.shape[0]/DELAY), 5, B)) # 290 = lenght test dataset divided by delay

    ## Test Model with test dataset

    for idx, inputs in enumerate(val_dataset):
        print(idx, inputs.shape)
        print(f'B: {B}')
        for i in range(B):
            # if idx < 2:
            if inputs.shape[0] == 128:
                load_forecasted[idx * 128: (idx+1)*128, :, i] = model.predict(inputs,verbose=0)
            else:
                load_forecasted[-inputs.shape[0]:, :, i] = model.predict(inputs,verbose=0)

    load_forecasted = np.repeat(load_forecasted, DELAY, axis=0)

    if not os.path.exists(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/Th_{DELAY}/'):
        os.makedirs(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/Th_{DELAY}/')

    np.save(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/Th_{DELAY}/training_block1_delay_{DELAY}_phi_{PHI}_gamma_{GAMMA}_deltax_005.npy',
    load_forecasted)

    static_val = np.load(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/Th_{DELAY}/training_block1_delay_{DELAY}_phi_{PHI}_gamma_{GAMMA}_deltax_005.npy')

    ppf_val_list=[0.7,0.8,0.85,0.9,0.95,0.99]
    # ppf_val_list=[0.9]
    min_val_loss = 1e6
    for ppf_val in ppf_val_list:
        
        #Uncertainty propagation
        upper_load_forecasted=montecarlo_stuff(static_val,ppf_val)

        ## compute the validation loss
        val_loss= denorm_validation_loss(upper_load_forecasted,x_val_norm_towindow , minmaxscaler,DELAY)

        ## take the minimum validation loss depending on the ppf_val
        min_val_loss = min(val_loss, min_val_loss)

        # print (f' ppf_val : {ppf_val} with val loss: {val_loss} for PHI: {PHI}, DELAY: {DELAY}, Simulation: {Simulation}')

        if val_loss == min_val_loss:
            min_ppf_val = ppf_val
            print(f'new min val loss: {val_loss} for ppf_val: {min_ppf_val} , simulation: {Simulation}')
        ## take the minimum validation loss 
        

    print (f'Validation loss: {min_val_loss} for PHI: {PHI}, ETA: {ETA}, DELAY: {DELAY}, Simulation: {Simulation} with ppf_val: {min_ppf_val}')
    
    return min_val_loss
  

def Diagram_Block_1_Training_scipy(config,TB_Fpath,save=True):
    global city, XTRAIN, LOOKBACK, DELAY, PHI, GAMMA, NUM_SERV, EPOCHS, B, Simulation,ppf_helper, ETA, TBPATH
    print('Using scipy')
    ## Set Variables
    city = config['city']
    #XTRAIN = config['XTRAIN']
    LOOKBACK = config['LOOKBACK']
   # DELAY = config['DELAY_Block1_Block2']
    global PHI 
    PHI = config['PHI']
    global GAMMA
    GAMMA = config['GAMMA']
    NUM_SERV = config['NUM_SERV_Block_1']
    EPOCHS = config['EPOCHS_block1']
    B = config['B']
    Simulation = config['Simulations']
    ppf_helper = config['ppf_helper']
   
    TBPATH=TB_Fpath

    ETA = config['ETA']

    window_min=2 # starting lower bound
    # window_max=120 # starting upper bound (previously 120)
    window_max=100 # starting upper bound (previously 120)
    if not os.path.exists (f'/home/sergi_alcala/sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/optimal_window.npy'):
        val_loss=scipy.optimize.minimize_scalar(Training_Block, bounds=(window_min, window_max), method='bounded')
        print (f'val loss value:{val_loss.fun}')
        print (f'val loss window:{val_loss.x}')
        
        np.save(f'./sergi_data/AZTEC_extension/{TBPATH}/{city}/PHI_{PHI}/ETA_{ETA}/Simulation_{Simulation}/optimal_window.npy', val_loss.x)
    else:
        print ('Optimal window already exists')

