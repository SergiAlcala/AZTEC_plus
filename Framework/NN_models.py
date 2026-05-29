import numpy as np
import pandas as pd
import os, os.path,sys
from sklearn.preprocessing import MinMaxScaler
import tqdm
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras import layers, Input
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt





class NN_Models:
    def make_model_lstm(input_shape, lookback, num_services):
        inputs = Input(shape=(lookback, num_services))
        model = layers.LSTM(128)(inputs)
        #model = layers.LSTM(32)(model)
        #model = layers.Flatten()(model)
        #model = layers.Dense(64, activation='relu')(model)
        model = layers.Dropout(0.3)(model, training=True)
        model = layers.Dense(64, activation='relu')(model)
        model = layers.Dropout(0.3)(model, training=True)
        output = layers.Dense(num_services)(model)
        model = Model(inputs, output)
        #model.compile(optimizer=Adam(0.0005), loss=cost_func)
        return model

    def make_model_cnn(input_shape, lookback, num_services):
        inputs = Input(shape=(lookback, num_services))
        model = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        model = layers.Conv1D(32, 3, activation='relu', padding='same')(model)
        model = layers.Flatten()(model)
        model = layers.Dense(64, activation='relu')(model)
        model = layers.Dropout(0.3)(model, training=True)
        model = layers.Dense(32, activation='relu')(model)
        model = layers.Dropout(0.3)(model, training=True)
        output = layers.Dense(num_services)(model)
        model = Model(inputs, output)
        return model

    def make_model_lstm_block2(lookback):
        inputs = Input((lookback, 1))
        model = layers.LSTM(128)(inputs)
        #model = layers.Dense(64)(model)
        model = layers.Dropout(0.3)(model, training=True) ### original 0.2
        output = layers.Dense(1)(model)
        model = Model(inputs, output)
        return model