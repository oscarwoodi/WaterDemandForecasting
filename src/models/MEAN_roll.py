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


    def meantime(self, df, dma, stat='mean', res_window=24, p=0.95, residual=True):
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
        
        return avg_df[['mean']], avg_df[['upper_conf']], avg_df[['lower_conf']]


    def roll_predict(self, no_test, no_batch): 
        """
        apply rolling validtaion.
        
        no_test - number of days of data in entire test set
        no_batch - number of days of data in each test batch

        """
        # iteration 1
        epoch_train = self.train
        epoch_test = self.test.iloc[:24*no_batch, :]
        self.roll_results = pd.DataFrame(columns=self.test.columns)
        
        epoch_result = self.predict(epoch_train, epoch_test)
        self.roll_results = pd.concat([self.roll_results, epoch_result])
        
        for i in np.arange(0, no_test/no_batch-1): 
            epoch_train = pd.concat([epoch_train, self.test.iloc[int(i*24*no_batch):int((i+1)*24*no_batch), :]])
            epoch_test = self.test.iloc[int((i+1)*24*no_batch):int((i+2)*24*no_batch), :] 

            epoch_result = self.predict(epoch_train, epoch_test)
            self.roll_results = pd.concat([self.roll_results, epoch_result])

        self.roll_results.index.name = 'date_time'

        return self.roll_results

    def predict(self, train, test):
        """
        Make predictions using the trained ARIMA model.
        """
        results = test.copy()

        # iterate over dmas
        for dma in self.dmas: 
            print(f"Running for {dma}...")
            forecast_df = pd.DataFrame(columns=test.columns, index=test.index)
            result_dma = pd.concat([train, forecast_df], axis=0)
            
            forecast, upper, lower = self.meantime(
                result_dma, 
                dma,  
                res_window=20, 
                p=0.95, 
                residual=True)
            
            # extract forecast
            results[dma] = forecast.iloc[-len(test):, :]
            results['lower_'+dma] = lower.iloc[-len(test):, :]
            results['upper_'+dma] = upper.iloc[-len(test):, :]

        return results
    
    def save(self, filepath):
        """
        Save model results to csv.
        """
        self.roll_results.to_csv(filepath)
        

# Example usage:
if __name__ == "__main__":
    arima_model = MEANModel()
    arima_model.import_data(train_window=1*7, test_window=7, filepath_demand="../../data/InflowDataImputedNoanomoly.csv", filepath_weather="../../data/WeatherDataImputed.csv", filepath_diurnal="../../data/DiurnalDataImputed.csv")
    predictions = arima_model.roll_predict(7, 1)
    arima_model.save(filepath="../../results/MEAN_1W_+24H.csv")