import pandas as pd
import numpy as np
    
def mape(actuals, predictions):
    """Mean absolute percentage error"""
    return (abs(predictions - actuals) / actuals).mean()

def rmse(actuals, predictions):
    """Root mean squared error"""
    return (((predictions - actuals)**2).mean())**0.5

def mae(actuals, predictions):
    """Mean absolute error"""
    return abs(predictions - actuals).mean()

def r(actuals, predictions): 
    """r-squared error"""
    ssreg = numpy.sum((actuals - prediction)**2)
    sstot = numpy.sum((actuals - actuals.mean())**2)
    
    return 1 - ssreg / sstot

#Evaluate all
def scores(test,pred):
    score_rmse = rmse(test,pred)
    score_mape = mape(test,pred)
    score_mae = mae(test,pred)
    #score_nse = nse(test,pred)
    df = pd.DataFrame({'indicator':['RMSE','MAPE','MAE'],
          'value':[score_rmse,score_mape,score_mae]})
    df.set_index('indicator',inplace=True)
    return (df)
