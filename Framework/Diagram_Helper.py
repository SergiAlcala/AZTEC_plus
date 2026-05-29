from Libraries import *




def Diagram_Helper(config,save_folder,save=True):

    ## Set Variables
    city=config['city']
    LOOKBACK=config['LOOKBACK']
    DELAY=config['DELAY_Helper']
    NUM_SERV_B1=config['NUM_SERV_Block_1']
    NUM_SERV=config['NUM_SERV_Helper']
    EPOCHS=config['EPOCHS_Helper']
    B=config['B']
    SEL_SERV_List=config['SEL_SERV_H']

    check_exist_fpath= f'./sergi_data/AZTEC_extension/{save_folder}/{city}/helper_forecasting_delay_{DELAY}.npy'

    
    if not os.path.exists(check_exist_fpath):

        ## Load Data

        bordeaux = pd.read_csv(f'./sergi_data/AZTEC_extension/citys/{city}.csv')
        bordeaux.drop('date_time', axis=1, inplace=True)
        bordeaux = bordeaux.reindex(sorted(bordeaux.columns), axis=1)
        bordeaux = bordeaux.to_numpy()

        XTRAIN=round(len(bordeaux)*0.8)



        ## Normalize Data


        minmaxscaler = MinMaxScaler()
        x_train = bordeaux[:XTRAIN]
        x_train_norm = tf.convert_to_tensor(minmaxscaler.fit_transform(x_train), dtype=tf.float32)
        x_test = bordeaux[XTRAIN:]
        x_test_norm = tf.convert_to_tensor(minmaxscaler.transform(x_test), dtype=tf.float32)

        test_load_forecasted_merged = np.zeros((x_test.shape[0]-LOOKBACK, B, NUM_SERV_B1))
        train_load_forecasted_merged = np.zeros((x_train.shape[0]-LOOKBACK, B, NUM_SERV_B1))

        ## Prepare Data
        for SEL_SERV in SEL_SERV_List:


            print (f'City: {city} - Service: {SEL_SERV}')

            # For one service only
            input_dataset = keras.preprocessing.timeseries_dataset_from_array(x_train_norm[:-DELAY, SEL_SERV], None,
                                                                                sequence_length=LOOKBACK, sequence_stride=DELAY)
            target_dataset = keras.preprocessing.timeseries_dataset_from_array(x_train_norm[LOOKBACK:, SEL_SERV], None,
                                                                                sequence_length=DELAY, sequence_stride=DELAY)
            test_dataset = keras.preprocessing.timeseries_dataset_from_array(x_test_norm[:-DELAY, SEL_SERV], None,
                                                                                sequence_length=LOOKBACK, sequence_stride=DELAY,
                                                                                batch_size=128)

            ## Load Model

            model = NN.make_model_lstm(x_train_norm.shape, LOOKBACK, NUM_SERV)
            optimizer = Adam(learning_rate=0.0005)
            mae = tf.keras.losses.MeanAbsoluteError()

            ## Train Model
            for epoch in tqdm(range(EPOCHS)):
                #print("\nStart of epoch %d" % (epoch,))
                for step, (x_batch_train, y_batch_train) in enumerate(zip(input_dataset, target_dataset)):
                    with tf.GradientTape() as tape:
                        prediction = model(x_batch_train, training=True)
                        loss_value = mae(y_batch_train, prediction)

                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # FOR ONE SERVICE FOR TEST FORECASTING

            load_forecasted = np.zeros((x_test.shape[0]-LOOKBACK, B))

            ## Test Model for Test Data

            for idx, inputs in enumerate(test_dataset):
                print(idx, inputs.shape)
                for i in range(B):
                    if inputs.shape[0] == 128:
                        load_forecasted[idx*128:(idx+1)*128, i] = model.predict(inputs,verbose=0)[:,0]
                       
                    else:
                         load_forecasted[-inputs.shape[0]:, i] = model.predict(inputs,verbose=0)[:,0]


            load_forecasted = np.repeat(load_forecasted, DELAY, axis=0)

            test_load_forecasted_merged[:, :, SEL_SERV] = load_forecasted

            print(f'Test Service {SEL_SERV} finished for {city}')
            

            ## Test Model for Training Data
            # FOR ONE SERVICE FOR TRAINING FORECASTING

            load_forecasted = np.zeros((x_train.shape[0]-LOOKBACK, B))

            for idx, inputs in enumerate(input_dataset):
                print (f'Train idx {idx}, xtest_shape: {inputs.shape[0]}')
                for i in range(B):
                    if idx == x_train.shape[0] // 128:
                        load_forecasted[-inputs.shape[0]:, i] = model.predict(inputs,verbose=0)[:,0]
                    else:
                        load_forecasted[idx*128:(idx+1)*128, i] = model.predict(inputs,verbose=0)[:,0]

            load_forecasted = np.repeat(load_forecasted, DELAY, axis=0)


            train_load_forecasted_merged[:, :, SEL_SERV] = load_forecasted
            print(f'Train Service {SEL_SERV} finished for {city}')

        ## Save Results
        if save == True:
            if not os.path.exists(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/'):
                os.makedirs(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/')

            np.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/helper_forecasting_delay_{DELAY}.npy', test_load_forecasted_merged)
            np.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/helper_training_forecasting_delay_{DELAY}.npy', train_load_forecasted_merged)
            model.save(f'./sergi_data/AZTEC_extension/{save_folder}/{city}/helper_model_delay_{DELAY}.h5')
        print (f'Helper Model for {city} with delay {DELAY} Finished')
    else:
        print (f'Helper Model for {city} with delay {DELAY} already exists')
