import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import time as time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, Dropout, GRU, SimpleRNN, Concatenate
from tensorflow.keras.models import Model, Sequential

from utils.utils import extract_features

class ANNModel:
    """
    ANN model
    """
    def __init__(self):
        self.model = None
    
    def import_data(self, train_window, filepath_demand, filepath_weather, filepath_diurnal):
        """
        Import data from CSV file.
        """
        # load imputed inflow data
        self.inflow_data = pd.read_csv(filepath_demand)
        self.inflow_data = self.inflow_data.set_index('date_time')
        self.inflow_data.index = pd.to_datetime(self.inflow_data.index)

        # load diurnal flow data
        self.diurnal_inflow_data = pd.read_csv(filepath_diurnal)
        self.diurnal_inflow_data = self.diurnal_inflow_data.set_index('date_time')
        self.diurnal_inflow_data.index = pd.to_datetime(self.diurnal_inflow_data.index)

        # load weather data
        self.weather_data = pd.read_csv(filepath_weather)
        self.weather_data = self.weather_data.set_index('date_time')
        self.weather_data.index = pd.to_datetime(self.weather_data.index)

        # make list with dma names
        self.dmas = list(self.inflow_data.columns)

        # remove dupes
        self.inflow_data = self.inflow_data[~self.inflow_data.index.duplicated(keep='first')]
        self.weather_data = self.weather_data[~self.weather_data.index.duplicated(keep='first')]
        self.inflow_diurnal_data = self.diurnal_inflow_data[~self.diurnal_inflow_data.index.duplicated(keep='first')]

        # splitting data
        self.inflow_df = self.inflow_data[(self.inflow_data.index >= self.inflow_data.index[-1]-timedelta(weeks=train_window+1))].copy()
        self.inflow_diurnal_df = self.diurnal_inflow_data[(self.diurnal_inflow_data.index >= self.diurnal_inflow_data.index[-1]-timedelta(weeks=train_window+1))].copy()
        self.weather_df = self.weather_data[(self.weather_data.index >= self.weather_data.index[-1]-timedelta(weeks=train_window+1))].copy()

        # extract time features
        self.inflow_df = extract_features(self.inflow_df)

    def split(self, test_window=168, validation_window=168): 
        """
        Split data into train, validation and test.
        """
        # using one week for testing period (hourly resolution)
        no_train = len(self.inflow_df) - 168
        no_test = test_window
        no_vali = validation_window

        # split into train and test
        self.train = self.inflow_df.iloc[0:no_train, :][self.dmas].copy()
        self.test = self.inflow_df.iloc[no_train:, :][self.dmas].copy()

        self.train_diurnal = self.inflow_diurnal_df.iloc[0:no_train, :][self.dmas].copy()
        self.test_diurnal = self.inflow_diurnal_df.iloc[no_train:, :][self.dmas].copy()

        self.train_exog = self.weather_df.iloc[0:no_train, :].copy()
        self.test_exog = self.weather_df.iloc[no_train:no_train+no_test, :].copy()

    def transform(self, df, scalers=None, inverse=False): 
        """
        Applies and inverses Box-cox tranformation to data
        """
        # apply box-cox transform
        if not inverse: 
            scalers = {dma: None for dma in df.columns}
            for col in df.columns:
                x = df[col].values.astype('float32')
                sc = StandardScaler()
                sc = sc.fit(x.reshape(-1,1))
                standardised = sc.transform(x.reshape(-1,1))

                self.inflow_df[col] = standardised
                scalers[col] = sc
                
            return scalers
        
        # inverse box-cox transform 
        elif inverse: 
            for col in df.columns:
                x = df[col]
                sc = scalers[col]
                original = sc.inverse_transform(x)
                self.inflow_df[col] = original
    
    def build_model(hp):
        dropout = hp.Float('dropout_rate',min_value=0.1,max_value=0.5,step=0.05,default=0.2)

        #Time Inputs
        time_input = Input(shape=(X_train.shape[1], ),name="time")
        
        time_features = Dense(hp.Int('nodes_input',min_value = 32,max_value = 256, step=32), activation = 'relu')(time_input)
        time_features = Dense(hp.Int('nodes_1',min_value = 32,max_value = 256, step=32), activation = 'relu')(time_features)
        time_features = Dropout(dropout)(time_features)

        #Exog Inputs
        exog_input = Input(shape=(X_train_exog.shape[1], ),name="exog")
        
        #Merge exog and x
        x = Concatenate(axis=1)([time_features,exog_input])

        #Dense layers for more feature extraction
        x = Dense(hp.Int('nodes_2',min_value = 32,max_value = 256, step=32), activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(hp.Int('nodes_3',min_value = 32,max_value = 256, step=32), activation='relu')(x)

        #Gaussian Distribution
        outputs = Dense(out_size*2)(x)
        distribution_outputs = Lambda(gaussian_distribution_layer)(outputs)
        
        #Define model
        model = Model(inputs=[time_input,exog_input], outputs=distribution_outputs)

        #Tune learning rate from 0.0001 to 0.01
        #opt = Adam(hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log'))
        #Hyperbolic decrease learning rate to prevent overfitting, tune initial learning rate
        opt = Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                        hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log'),
                        decay_steps=train_size*1000,
                        decay_rate=1,
                        staircase=False),
                        clipvalue=0.5,
                        clipnorm=1
                        )
        
        model.compile(loss = gaussian_loss, optimizer = opt)

        return model
