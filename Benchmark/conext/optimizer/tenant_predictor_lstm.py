#!/usr/bin/python
import sys

import os
import numpy as np
import math
from helpers import *
import time

from pandas import DataFrame
from pandas import Series
from pandas import concat
#from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_json

WARM_UP = 2
TRAINING_PERIOD = 5

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]#!/usr/bin/python
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
#def scale(train, test):
    ## fit scaler
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = scaler.fit(train)
    ## transform train
    #train = train.reshape(train.shape[0], train.shape[1])
    #train_scaled = scaler.transform(train)
    ## transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    #test_scaled = scaler.transform(test)
    #return scaler, train_scaled, test_scaled
def scale(train):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)

    return scaler, train_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


def make_forecast(raw_values, myrequest, mytenant, Tdec):
    mybs = 1
    ## compute lambda_hat   
    if len(raw_values) < 2:
        lambda_hat = myrequest
        sigma_hat = 0
        return lambda_hat, sigma_hat
    
    ## compute lambda_hat    
    n_intervals = len(raw_values)/Tdec
    # get sample windows (stats over each decision interval)
    offset = 0
    samples = []
    for k in range(n_intervals):
        samples.append(np.max(raw_values[offset:offset+Tdec])) #stat used is mean() TBD: do we wanna change the stat?
        offset = offset + Tdec
        
    if n_intervals < WARM_UP:
        lambda_hat = myrequest
    else:        
    
        # transform data to be stationary
        diff_values = difference(samples, 1)

        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values

        # TBD: Training can be done from time to time in a separate process -- it should not be done everytime!!
        
        train = supervised_values
        # split data into train and test-sets
        #train, test = supervised_values[0:-12], supervised_values[-12:]

        # transform the scale of the data
        scaler, train_scaled = scale(train)

        if (n_intervals == WARM_UP) or (n_intervals % TRAINING_PERIOD == 0): #train the model!! #TBD: WRONG. SAVE PER TENANT AND BS!!!
            print("LSTM (re)tranining model...")
            # fit the model
            lstm_model = fit_lstm(train_scaled, 1, 500, 4)
            
            # forecast the entire training dataset to build up state for forecasting
            train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
            lstm_model.predict(train_reshaped, batch_size=1)
            
            # serialize model to JSON
            model_json = lstm_model.to_json()
            model_name = "lst_model" + "_" + str(mytenant) + "_" + str(mybs) 
            with open("/tmp/" + model_name + ".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            lstm_model.save_weights("/tmp/" + model_name + ".h5")
            print("Saved model to disk")
                
                
        if (n_intervals > WARM_UP) and (n_intervals % TRAINING_PERIOD != 0): # load model if it is not fresh
            model_name = "lst_model" + "_" + str(mytenant) + "_" + str(mybs)             
            json_file = open("/tmp/" + model_name + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            lstm_model = model_from_json(loaded_model_json)
            # load weights into new model
            lstm_model.load_weights("/tmp/" + model_name + ".h5")
            print("Loaded model from disk") 
            
        # make one-step forecast
        #print(train_scaled)
        i = 0 # just next value
        X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
        lambda_hat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        lambda_hat = invert_scale(scaler, X, lambda_hat)
        # invert differencing
        lambda_hat = min(myrequest, inverse_difference(samples, lambda_hat, len(train_scaled)+1-i))
        
    
    
    ## compute sigma_hat
    sigma_hat = myrequest - lambda_hat # maximum uncertainty
    if n_intervals > 1:
        samples2 = []
        for k in range(len(samples)):
            samples2.append(max(0, samples[k]-lambda_hat))
            
        sigma_hat = max(1, np.average(samples2))
        
        
    
    
    return lambda_hat, sigma_hat



            
            