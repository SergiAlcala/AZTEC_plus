from Libraries import *
from Diagram_Call import *



def Diagram_Block_2(config,save_folder,save=True):


    city=config['city'] 
    LOOKBACK=config['LOOKBACK']
    DELAY=config['DELAY_Block1_Block2']  
    PHI=config['PHI']
    GAMMA=config['GAMMA']
    EPOCHS=config['EPOCHS_block2']
    B=config['B']
    ALPHA=config['ALPHA']
    ppf_helper=config['ppf_helper']
    ppf_static=config['ppf_static']
    ETA = config['ETA']
    
    DIR = '.'

    if  not os.path.exists(f'{DIR}/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{DELAY}/shared_fore_uncer_block2_delay_{DELAY}_phi_{PHI}_alpha_{ALPHA}_gamma_2.npy'):
        def cost_func(y_true, y_pred, alpha, delay, gamma):
            step = alpha
            forecast_delay = delay
            #delta_x = 0.05
            epsilon = -step / 0.1
            y_pred = tf.expand_dims(y_pred, axis=1)
            y_pred = tf.tile(y_pred, (1, forecast_delay, 1))
            cost = y_pred - y_true# - delta_x
            big_penalty = epsilon * (cost + 0.1) + step
            penalty = -0.1 * (cost + 0.1) + step
            pen_positive = cost * gamma
            cost = tf.where(cost > 0, pen_positive, cost)
            cost = tf.where(cost < -0.1, penalty, cost)
            cost = tf.where(tf.logical_and((cost <= 0), (cost >= -0.1)),
                            big_penalty, cost)
            cost = tf.abs(cost)
            cost = K.sum(K.sum(cost, axis=-1), axis=-1)
            cost= K.mean(cost) ## ADDED NOW, Maybe remove it
            return cost

            ## Load Data
        bordeaux = pd.read_csv(f'{DIR}/sergi_data/AZTEC_extension/citys/{city}.csv')
        bordeaux.drop('date_time', axis=1, inplace=True)
        bordeaux = bordeaux.reindex(sorted(bordeaux.columns), axis=1)
        bordeaux = bordeaux.to_numpy()

            ##Load Normalization
        normalizator = pickle.load(open(f'{DIR}/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}/block_1_minmaxscaler.pkl', 'rb'))

            ##Transpose Data
        helper_forecasting_training = np.transpose(
            np.load(f'{DIR}/sergi_data/AZTEC_extension/{save_folder}/{city}/helper_training_forecasting_delay_1.npy'), (0, 2, 1))
        helper_forecasting_test = np.transpose(np.load(f'{DIR}/sergi_data/AZTEC_extension/{save_folder}/{city}/helper_forecasting_delay_1.npy'),
                                            (0, 2, 1))

        # MinMaxScaler normalize between 0 and 1
        helper_forecasting_training = helper_forecasting_training.clip(min=0)
        helper_forecasting_test = helper_forecasting_test.clip(min=0)

        dist_help_training = scipy.stats.norm(loc=helper_forecasting_training.mean(axis=-1),
                                            scale=helper_forecasting_training.std(axis=-1))
        dist_help_test = scipy.stats.norm(loc=helper_forecasting_test.mean(axis=-1),
                                        scale=helper_forecasting_test.std(axis=-1))

        upper_help_training = dist_help_training.ppf(ppf_helper) ## Remember that you can modify these values
        upper_help_test = dist_help_test.ppf(ppf_helper) ## Remember that you can modify these values

        upper_help_training[np.where(np.isnan(upper_help_training))] = (
            helper_forecasting_training[np.where(np.isnan(upper_help_training))][:, 1])
        upper_help_test[np.where(np.isnan(upper_help_test))] = (
            helper_forecasting_test[np.where(np.isnan(upper_help_test))][:, 1])

        
        

        static_training = np.load(f'{DIR}/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}/cap_fore_uncer_training_block1_delay_{DELAY}_phi_{PHI}_gamma_2_deltax_005.npy')
        static_test = np.load(f'{DIR}/sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Th_{DELAY}/cap_fore_uncer_test_block1_delay_{DELAY}_phi_{PHI}_gamma_2_deltax_005.npy')

        # MinMaxScaler normalize between 0 and 1
        static_training = static_training.clip(min=0)
        static_test = static_test.clip(min=0)

        dist_static_training = scipy.stats.norm(loc=static_training.mean(axis=-1),
                                                scale=static_training.std(axis=-1))
        dist_static_test = scipy.stats.norm(loc=static_test.mean(axis=-1),
                                            scale=static_test.std(axis=-1))
        
        upper_static_training = dist_static_training.ppf(ppf_static) ## Remember that you can modify these values
        upper_static_test = dist_static_test.ppf(ppf_static) ## Remember that you can modify these values


        upper_static_training[np.where(np.isnan(upper_static_training))] = (
            static_training[np.where(np.isnan(upper_static_training))][:, 0])
        upper_static_test[np.where(np.isnan(upper_static_test))] = (
            static_test[np.where(np.isnan(upper_static_test))][:, 0])
        
        ######## Align the datasets ######
        ### for training dataset


        diff = min(upper_static_training.shape[0], upper_help_training.shape[0])
        upper_static_training_diff = upper_static_training.shape[0] - diff
        upper_help_training_diff = upper_help_training.shape[0] - diff




        upper_static_training_diff = upper_static_training.shape[0] - diff
        upper_help_training_diff = upper_help_training.shape[0] - diff




        if upper_static_training_diff != 0:
            upper_static_training = upper_static_training[upper_static_training_diff:]
        if upper_help_training_diff != 0:
            upper_help_training = upper_help_training[upper_help_training_diff:]

        diff = min(upper_static_test.shape[0], upper_help_test.shape[0])
        upper_static_test_diff = upper_static_test.shape[0] - diff
        upper_help_test_diff = upper_help_test.shape[0] - diff


        if upper_static_test_diff != 0:
            upper_static_test = upper_static_test[upper_static_test_diff:]
        if upper_help_test_diff != 0:
            upper_help_test = upper_help_test[upper_help_test_diff:]
        # if upper_help_training.shape[0]>upper_static_training.shape[0]:
        #     diff=upper_help_training.shape[0]-upper_static_training.shape[0]
        #     shared_capacity_training = (upper_help_training[diff:] - upper_static_training).clip(min=0)

        # elif upper_help_training.shape[0]<upper_static_training.shape[0]:
        #     diff=upper_static_training.shape[0]-upper_help_training.shape[0]
        #     shared_capacity_training = (upper_help_training - upper_static_training[diff:]).clip(min=0)
            
        # else:
        #     shared_capacity_training = (upper_help_training - upper_static_training).clip(min=0)

        # for test dataset
        
        # if upper_help_test.shape[0]>upper_static_test.shape[0]:
        #     diff=upper_help_test.shape[0]-upper_static_test.shape[0]
        #     shared_capacity_test = (upper_help_test[diff:] - upper_static_test).clip(min=0)

        # elif upper_help_test.shape[0]<upper_static_test.shape[0]:
        #     diff=upper_static_test.shape[0]-upper_help_test.shape[0]
        #     shared_capacity_test = (upper_help_test - upper_static_test[diff:]).clip(min=0)

        # else:
        #     shared_capacity_test = (upper_help_test - upper_static_test).clip(min=0) 
        
        shared_capacity_test = (upper_help_test - upper_static_test).clip(min=0)  
        shared_capacity_training = (upper_help_training - upper_static_training).clip(min=0)


        shared_capacity_training_den = normalizator.inverse_transform(shared_capacity_training)
        shared_capacity_test_den = normalizator.inverse_transform(shared_capacity_test)
            
        ########################################

        agg_minmaxscaler = MinMaxScaler()
        agg_shared_capacity_training = tf.convert_to_tensor(agg_minmaxscaler.fit_transform(shared_capacity_training_den.sum(axis=-1).reshape(-1, 1)),
                                                            dtype=tf.float32)
        agg_shared_capacity_test = tf.convert_to_tensor(agg_minmaxscaler.transform(shared_capacity_test_den.sum(axis=-1).reshape(-1, 1)),
                                                    dtype=tf.float32)
        if save:
            if not os.path.exists(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{DELAY}'):
                os.makedirs(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{DELAY}')
            pickle.dump(agg_minmaxscaler, open(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{DELAY}/block_2_minmaxscaler.pkl', 'wb'))

        input_dataset = keras.preprocessing.timeseries_dataset_from_array(agg_shared_capacity_training[:-DELAY], None,
                                                                        sequence_length=LOOKBACK, sequence_stride=DELAY)
        target_dataset = keras.preprocessing.timeseries_dataset_from_array(agg_shared_capacity_training[LOOKBACK:], None,
                                                                        sequence_length=DELAY, sequence_stride=DELAY)

        test_dataset = keras.preprocessing.timeseries_dataset_from_array(agg_shared_capacity_test[:-DELAY], None,
                                                                        sequence_length=LOOKBACK, sequence_stride=DELAY,
                                                                        batch_size=128)
        model = NN.make_model_lstm_block2(LOOKBACK)

        optimizer = Adam(learning_rate=0.0005)

        
        print(f' City: {city} - Phi: {PHI} - Alpha: {ALPHA} - Delay: {DELAY} - Gamma: {GAMMA} - Lookback: {LOOKBACK}')
        for epoch in tqdm(range(EPOCHS)):
            #print("\nStart of epoch %d" % (epoch,))
            for step, (x_batch_train, y_batch_train) in enumerate(zip(input_dataset, target_dataset)):
                with tf.GradientTape() as tape:
                    prediction = model(x_batch_train, training=True)
                    loss_value = cost_func(y_batch_train, prediction, ALPHA, DELAY, GAMMA)
                    
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

        load_forecasted = np.zeros((int(agg_shared_capacity_test.shape[0]/DELAY), B))

        for idx, inputs in enumerate(test_dataset):
            for i in range(B):
                if inputs.shape[0] == 128:
                    load_forecasted[idx * 128: (idx+1)*128, i] = model.predict(inputs,verbose=0)[:, 0]
                else:
                    load_forecasted[-inputs.shape[0]:, i] = model.predict(inputs,verbose=0)[:, 0]



         
        load_forecasted = np.repeat(load_forecasted, DELAY, axis=0)

        if save:
            np.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{DELAY}/shared_fore_uncer_block2_delay_{DELAY}_phi_{PHI}_alpha_{ALPHA}_gamma_2.npy', load_forecasted)
            model.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/PHI_{PHI}/ETA_{ETA}/Block_2_3_results/ppf_static_{ppf_static}_ppf_helper_{ppf_helper}/Th_{DELAY}/block_2_model_delay_{DELAY}_phi_{PHI}_alpha_{ALPHA}_gamma_2.h5')
        print('Block 2 finished')

    else:
        print('Block 2 skipped')
        
