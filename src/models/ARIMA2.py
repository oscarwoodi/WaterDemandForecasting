# imports
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs
from datetime import timedelta

class ARIMAModel:
    """
    ARIMA model on demand - dirunal demand. 
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

        # using one week for testing period (hourly resolution)
        no_train = len(self.inflow_df) - 168
        no_test = 168

        # split into train and test
        self.train = self.inflow_df.iloc[0:no_train, :][self.dmas].copy()
        self.test = self.inflow_df.iloc[no_train:, :][self.dmas].copy()

        self.train_diurnal = self.inflow_diurnal_df.iloc[0:no_train, :][self.dmas].copy()
        self.test_diurnal = self.inflow_diurnal_df.iloc[no_train:, :][self.dmas].copy()

        self.train_exog = self.weather_df.iloc[0:no_train, :].copy()
        self.test_exog = self.weather_df.iloc[no_train:no_train+no_test, :].copy()

    def find_best_parameters(self):
        """
        Find the best ARIMA parameters using AIC criterion.
        """
        # Perform grid search to find the best ARIMA parameters
        # run auto arima to find best model parameters
        self.arima_params = {dma: () for dma in self.dmas}

        for dma in self.dmas: 
            arima_d = ndiffs(self.train[dma]-self.train_diurnal[dma],max_d = 12)
            arima_model = auto_arima(self.train[dma]-self.train_diurnal[dma], d = arima_d, 
                                    seasonal=False, trace=True,
                                    error_action = 'ignore',
                                    suppress_warnings=True,stepwise= True,
                                    n_fits = 50, method = 'nm')

            self.arima_params[dma] = arima_model.get_params()['order']
            
        return self.arima_params
    
    def train_model(self, arima_params):
        """
        Train the ARIMA model using the best parameters.
        """
        models = {dma: np.nan for dma in self.dmas}
        for dma in self.dmas:
            print(f"Running for {dma}...")
            model_arima = ARIMA(self.train[dma]-self.train_diurnal[dma],
                                order = arima_params[dma], 
                                freq = None
                                )
            models[dma]=model_arima.fit()
        return models

    def predict(self, models):
        """
        Make predictions using the trained ARIMA model.
        """
        self.results = self.test.copy()
        self.ci_lower = pd.DataFrame(index=self.test.index)
        self.ci_upper = pd.DataFrame(index=self.test.index)
        
        for dma in self.dmas: 

            #Train prediction
            trainPredict_arima = (models[dma].predict(0,len(self.train)-1)).fillna(0)

            #Test Prediction
            forecast_arima = models[dma].get_prediction(len(self.train),len(self.inflow_df)-1)
            testPredict_arima = forecast_arima.predicted_mean+self.test_diurnal[dma]
            testPredict_arima_ci = forecast_arima.conf_int(alpha=0.1)
            testPredict_arima_ci.iloc[:,0] = testPredict_arima_ci.iloc[:,0]+self.test_diurnal[dma]
            testPredict_arima_ci.iloc[:,1] = testPredict_arima_ci.iloc[:,1]+self.test_diurnal[dma]

            # extract forecast
            self.results[dma] = testPredict_arima
            self.results['lower_'+dma] = testPredict_arima_ci['lower '+dma]
            self.results['upper_'+dma] = testPredict_arima_ci['upper '+dma]

        return self.results
    
    def save(self, filepath):
        """
        Save model results to csv.
        """
        self.results.to_csv(filepath)
        

# Example usage:
if __name__ == "__main__":
    arima_model = ARIMAModel()
    arima_model.import_data(train_window=4, filepath_demand="../../data/InflowDataImputed.csv", filepath_weather="../../data/WeatherDataImputed.csv", filepath_diurnal="../../data/DiurnalDataImputed.csv")
    params = arima_model.find_best_parameters()
    fit_models = arima_model.train_model(params)
    predictions = arima_model.predict(models=fit_models)
    arima_model.save(filepath="../../results/ARIMA2_4W.csv")