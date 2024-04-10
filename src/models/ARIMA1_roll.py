
# imports
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs
from datetime import timedelta

class ARIMAXModel:
    """
    ARIMA model on demand with dirunal as exogenous. 
    """
    def __init__(self):
        self.model = None
    
    def import_data(self, train_window, test_window, filepath_demand, filepath_weather, filepath_diurnal):
        """
        Import data from CSV file.

        train_window - number of days in train set
        test_window - number of days in test set

        """
        # load imputed inflow data
        self.inflow_data = pd.read_csv(filepath_demand)
        self.inflow_data = self.inflow_data.set_index('date_time')
        self.inflow_data.index = pd.to_datetime(self.inflow_data.index)
        self.inflow_data.index = self.inflow_data.index.to_period('H')

        # load diurnal flow data
        self.diurnal_inflow_data = pd.read_csv(filepath_diurnal)
        self.diurnal_inflow_data = self.diurnal_inflow_data.set_index('date_time')
        self.diurnal_inflow_data.index = pd.to_datetime(self.diurnal_inflow_data.index)
        self.diurnal_inflow_data.index = self.diurnal_inflow_data.index.to_period('H')\
        
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
        self.inflow_df = self.inflow_data[(self.inflow_data.index >= self.inflow_data.index[-1]-timedelta(days=train_window+test_window))].copy()
        self.inflow_diurnal_df = self.diurnal_inflow_data[(self.diurnal_inflow_data.index >= self.diurnal_inflow_data.index[-1]-timedelta(days=train_window+test_window))].copy()
        self.weather_df = self.weather_data[(self.weather_data.index >= self.weather_data.index[-1]-timedelta(days=train_window+test_window))].copy()

        # using one week for testing period (hourly resolution)
        no_train = len(self.inflow_df) - test_window*24
        no_test = test_window*24

        # split into train and test
        self.train = self.inflow_df.iloc[0:no_train, :][self.dmas].copy()
        self.test = self.inflow_df.iloc[no_train:, :][self.dmas].copy()

        self.train_diurnal = self.inflow_diurnal_df.iloc[0:no_train, :][self.dmas].copy()
        self.test_diurnal = self.inflow_diurnal_df.iloc[no_train:, :][self.dmas].copy()

        self.train_exog = self.weather_df.iloc[0:no_train, :].copy()
        self.test_exog = self.weather_df.iloc[no_train:no_train+no_test, :].copy()


    def roll_predict(self, no_test, no_batch): 
        """
        apply rolling validtaion.
        
        no_test - number of days of data in entire test set
        no_batch - number of days of data in each test batch

        """
        # iteration 1
        epoch_train = self.train
        epoch_train_diurnal = self.train_diurnal
        epoch_test = self.test.iloc[:24*no_batch, :]
        epoch_test_diurnal = self.test_diurnal.iloc[:24*no_batch, :]
        self.roll_results = pd.DataFrame(columns=self.test.columns)

        for i in np.arange(0, no_test/no_batch): 
            print(f"\n\nRunning validation step no. {int(i+1)}...\n\n")
        
            params = self.find_best_parameters(epoch_train, epoch_train_diurnal)
            fit_models = self.train_model(params, epoch_train, epoch_train_diurnal)  
            epoch_result = self.predict(fit_models, epoch_train, epoch_test, epoch_test_diurnal, 24*no_batch)
            self.roll_results = pd.concat([self.roll_results, epoch_result])
            
            epoch_train = pd.concat([epoch_train, self.test.iloc[int(i*24*no_batch):int((i+1)*24*no_batch), :]])
            epoch_train_diurnal = pd.concat([epoch_train_diurnal, self.test_diurnal.iloc[int(i*24*no_batch):int((i+1)*24*no_batch), :]])
            epoch_test = self.test.iloc[int((i+1)*24*no_batch):int((i+2)*24*no_batch), :] 
            epoch_test_diurnal = self.test_diurnal.iloc[int((i+1)*24*no_batch):int((i+2)*24*no_batch), :] 

        self.roll_results.index.name = 'date_time'

        return self.roll_results
            

    def find_best_parameters(self, train, train_diurnal):
        """
        Find the best ARIMA parameters using AIC criterion.
        """
        # Perform grid search to find the best ARIMA parameters
        # run auto arima to find best model parameters
        arima_params = {dma: () for dma in self.dmas}

        for dma in self.dmas: 
            arima_d = ndiffs(train[dma],max_d = 12)
            arima_model = auto_arima(train[dma], d = arima_d, 
                                    exogenous = train_diurnal[dma],
                                    seasonal=False, trace=True,
                                    error_action = 'ignore',
                                    suppress_warnings=True,stepwise= True,
                                    n_fits = 50, method = 'nm')

            arima_params[dma] = arima_model.get_params()['order']
            
        return arima_params
    
    def train_model(self, arima_params, train, train_diurnal):
        """
        Train the ARIMA model using the best parameters.
        """
        models = {dma: np.nan for dma in self.dmas}
        for dma in self.dmas:
            print(f"Running for {dma}...")
            model_arimax = ARIMA(train[dma],
                                order = arima_params[dma], 
                                exog = train_diurnal[dma]
                                )
            models[dma]=model_arimax.fit()
        return models

    def predict(self, models, train, test, test_diurnal, window):
        """
        Make predictions using the trained ARIMA model.
        """
        results = test.copy()
        
        for dma in self.dmas: 

            #Test Prediction
            forecast_arimax = models[dma].get_prediction(len(train),len(train)+window-1,exog=test_diurnal[dma])
            testPredict_arimax = forecast_arimax.predicted_mean
            testPredict_arimax_ci = forecast_arimax.conf_int(alpha=0.1)

            # extract forecast
            results[dma] = testPredict_arimax
            results['lower_'+dma] = testPredict_arimax_ci.iloc[:,0]
            results['upper_'+dma] = testPredict_arimax_ci.iloc[:,1]

        return results
    
    def save(self, filepath):
        """
        Save model results to csv.
        """
        self.roll_results.to_csv(filepath)
        
# Example usage:
if __name__ == "__main__":
    arima_model = ARIMAXModel()
    arima_model.import_data(train_window=4*7, test_window=4*7, filepath_demand="../../data/InflowDataImputed.csv", filepath_weather="../../data/WeatherDataImputed.csv", filepath_diurnal="../../data/DiurnalDataImputed.csv")
    arima_model.roll_predict(4*7, 7)
    arima_model.save(filepath="../../results/ARIMA1_4W_+168H.csv")
