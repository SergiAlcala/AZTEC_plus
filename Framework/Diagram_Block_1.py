from Libraries import *
from Diagram_Call import *



def Diagram_Block_1(config,save_folder,save=True):

    ## Set Variables
    city = config['city']
    
    LOOKBACK = config['LOOKBACK']
    DELAY = config['DELAY_Block1_Block2']
    PHI = config['PHI']
    GAMMA = config['GAMMA']
    NUM_SERV = config['NUM_SERV_Block_1']
    EPOCHS = config['EPOCHS_block1']
    B = config['B']
    ETA = config['ETA']

    check_exist_fpath=f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA{ETA}/Th_{DELAY}/cap_fore_uncer_test_block1_delay_{DELAY}_phi_{PHI}_gamma_{GAMMA}_deltax_005.npy'
    
    
    if not os.path.exists(check_exist_fpath):
        print(f'Block 1: {city} - {PHI} - {GAMMA} - {DELAY}')

        ## Load Data

        bordeaux = pd.read_csv(f'/home/sergi_alcala/sergi_data/AZTEC_extension/citys/{city}.csv')
        bordeaux.drop('date_time', axis=1, inplace=True)
        bordeaux = bordeaux.reindex(sorted(bordeaux.columns), axis=1)
        bordeaux = bordeaux.to_numpy()

        XTRAIN=round(len(bordeaux)*0.8) # 80% of the data is used for training

        ## Normalize Data
        minmaxscaler = MinMaxScaler()
        x_train = bordeaux[:XTRAIN]
        x_train_norm = tf.convert_to_tensor(minmaxscaler.fit_transform(x_train), dtype=tf.float32)
        x_test = bordeaux[XTRAIN:]
        x_test_norm = tf.convert_to_tensor(minmaxscaler.transform(x_test), dtype=tf.float32)

        print(f'lenght of x_train_norm: {len(x_train_norm)}, lenght of x_test_norm: {len(x_test_norm)}')

        ## Prepare Data

    
        input_dataset = keras.preprocessing.timeseries_dataset_from_array(x_train_norm[:-DELAY], None,
                                                                          sequence_length=LOOKBACK, sequence_stride=DELAY)
        target_dataset = keras.preprocessing.timeseries_dataset_from_array(x_train_norm[LOOKBACK:], None,
                                                                           sequence_length=DELAY, sequence_stride=DELAY)

    
        test_dataset = keras.preprocessing.timeseries_dataset_from_array(x_test_norm[:-DELAY], None,
                                                                         sequence_length=LOOKBACK, sequence_stride=DELAY,
                                                                         batch_size=128)


        print (f'City : {city}, PHI: {PHI}, DELAY: {DELAY}, GAMMA: {GAMMA}, ETA : {ETA} , B: {B}')

        ## Define Model and Loss Function
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
            cost= K.mean(cost) ###ADDED NOW TO AVERAGE OVER THE BATCH, maybe remove it
            return cost

        model =NN.make_model_lstm(x_train_norm.shape, LOOKBACK, NUM_SERV)
        optimizer = Adam(learning_rate=0.0005)

        ## Train Model
        for epoch in tqdm(range(EPOCHS)):
        #print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(zip(input_dataset, target_dataset)):
                with tf.GradientTape() as tape:
                    prediction = model(x_batch_train, training=True)
                    loss_value = cost_func(y_batch_train, prediction)

                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        #
        load_forecasted = np.zeros((int(x_test_norm.shape[0]/DELAY), NUM_SERV, B)) # 290 = lenght test dataset divided by delay
        #load_forecasted = np.zeros((290, 5, B)) # 290 = lenght test dataset divided by delay


        ## Test Model with test dataset

        for idx, inputs in enumerate(test_dataset):
            print(idx, inputs.shape)
            for i in range(B):
                #if idx < 2:
                if inputs.shape[0] == 128:
                    load_forecasted[idx * 128: (idx+1)*128, :, i] = model.predict(inputs,verbose=0)
                else:
                    load_forecasted[-inputs.shape[0]:, :, i] = model.predict(inputs,verbose=0)

        load_forecasted = np.repeat(load_forecasted, DELAY, axis=0)

        ## Save Results

        if save:    
            if not os.path.exists(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}'):
                os.makedirs(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}')
        
            np.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}/cap_fore_uncer_test_block1_delay_{DELAY}_phi_{PHI}_gamma_{GAMMA}_deltax_005.npy',
                load_forecasted)

        training_load_forecasted = np.zeros((int(x_train_norm.shape[0]/DELAY), NUM_SERV, B))

        ## Test Model with training dataset

        for idx, inputs in enumerate(input_dataset):
            #print(idx, inputs.shape)
            for i in range(B):
                if inputs.shape[0] == 128:
                    training_load_forecasted[idx * 128: (idx+1)*128, :, i] = model.predict(inputs,verbose=0)
                else:
                    training_load_forecasted[-inputs.shape[0]:, :, i] = model.predict(inputs,verbose=0)

        training_load_forecasted = np.repeat(training_load_forecasted, DELAY, axis=0)
        if save:
            np.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}/cap_fore_uncer_training_block1_delay_{DELAY}_phi_{PHI}_gamma_{GAMMA}_deltax_005.npy',
            training_load_forecasted)
            model.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}/block1_delay_{DELAY}_phi_{PHI}_gamma_{GAMMA}_deltax_005.h5')

            ## Save MinMaxScaler

            pickle.dump(minmaxscaler, open(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}/block_1_minmaxscaler.pkl', 'wb'))
        print('Block 1 finished')
    else:
        print(f'Block 1 Model for {city} with delay {DELAY} already exists')
