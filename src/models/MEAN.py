# imports
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm

class MEANModel:
    """
    SARIMA model on demand with diurnal as exogenous variable and daily seasonality. 
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

    def meantime(self, df, dma, stat='mean', res_window=20, p=0.95, residual=True):
        """
        Make predictions using mean day/hour and residuals.
        """
        # exponential mean component
        avg_df = df[[dma]].copy()
        avg_df['mean'] = 0
        avg_df['std_dev'] = 0
        avg_df['upper_conf'] = 0
        avg_df['lower_conf'] = 0
        
        # fill with nan mean for insufficient data for exponential
        avg_df['day'] = avg_df.index.weekday
        avg_df['hour'] = avg_df.index.hour
        
        if stat == 'mean': 
            avg_values = avg_df.groupby(by=['day', 'hour']).mean()[dma]
        elif stat == 'median': 
            avg_values = avg_df.groupby(by=['day', 'hour']).median()[dma]
            
        for idx, row in avg_df.iterrows(): 
            avg_df.loc[idx, 'mean'] = avg_values.loc[row.day, row.hour]
            
        std_vals = avg_df.groupby(by=['day', 'hour'])[[dma]].std()
        avg_df['std'] = [std_vals.loc[(x['day'],x['hour'])][0] for _,x in avg_df.iterrows()]
        
        if residual: 
            # residual correction component
            avg_df['residual'] = avg_df[dma] - avg_df['mean']
            avg_df['residual'] = avg_df['residual'].fillna(method='ffill')
            avg_df['res_correction'] = avg_df['residual'].rolling(window=res_window, min_periods=1).mean()
            avg_df['mean'] = avg_df['mean'] + avg_df['res_correction'].fillna(0)
        
        # set confidence bounds
        for i in avg_df.index: 
            ci = norm.interval(confidence=p, loc=avg_df.loc[i, 'mean'], scale=avg_df.loc[i, 'std'])
            avg_df.loc[i, 'upper_conf'] = ci[1]
            avg_df.loc[i, 'lower_conf'] = ci[0]
        
        return avg_df['mean'], avg_df['upper_conf'], avg_df['lower_conf']


    def predict(self):
        """
        Make predictions using the trained ARIMA model.
        """
        self.results = self.test.copy()

        # iterate over dmas
        for dma in self.dmas: 
            print(f"Running for {dma}...")
            forecast_df = pd.DataFrame(columns=self.test.columns, index=self.test.index)
            result_dma = pd.concat([self.train, forecast_df], axis=0)
            
            forecast, upper, lower = self.meantime(
                result_dma, 
                dma,  
                res_window=20, 
                p=0.95, 
                residual=True)
            
            # extract forecast
            self.results[dma] = forecast
            self.results['lower_'+dma] = lower
            self.results['upper_'+dma] = upper

        return self.results
    
    def save(self, filepath):
        """
        Save model results to csv.
        """
        self.results.to_csv(filepath)
        

# Example usage:
if __name__ == "__main__":
    arima_model = MEANModel()
    arima_model.import_data(train_window=8, filepath_demand="../../data/InflowDataImputed.csv", filepath_weather="../../data/WeatherDataImputed.csv", filepath_diurnal="../../data/DiurnalDataImputed.csv")
    predictions = arima_model.predict()
    arima_model.save(filepath="../../results/MEAN1.csv")