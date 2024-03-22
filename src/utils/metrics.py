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

def nse(actual, prediction): 
    """ NSE error """
    mean_observed = np.mean(actual)
    numerator = np.sum((actual - prediction) ** 2)
    denominator = np.sum((actual - mean_observed) ** 2)
    nse = 1 - (numerator / denominator)
    
    return nse
    
#Evaluate all
def scores(test, prediction):
    """
    for dma in test.columns: 
        score_rmse = rmse(test,pred)
        score_mape = mape(test,pred)
        score_mae = mae(test,pred)
        #score_nse = nse(test,pred)
        df = pd.DataFrame({'indicator':['RMSE','MAPE','MAE'],
              'value':[score_rmse,score_mape,score_mae]})
        df.set_index('indicator',inplace=True)
     """
    # pre-allocate
    dmas = test.columns
    scores = {}
    
    # get scores
    mape_result = {dma: mape(test[dma], prediction[dma]) for dma in dmas}
    mape_result['total'] = sum(mape_result.values())
    rmse_result = {dma: rmse(test[dma], prediction[dma]) for dma in dmas}
    rmse_result['total'] = sum(rmse_result.values())
    mae_result = {dma: mae(test[dma], prediction[dma]) for dma in dmas}
    mae_result['total'] = sum(mae_result.values())
    nse_result = {dma: nse(test[dma], prediction[dma]) for dma in dmas}
    nse_result['total'] = sum(nse_result.values())

    scores['mape'] = mape_result
    scores['rmse'] = rmse_result
    scores['mae'] = mae_result
    scores['nse'] = nse_result
    
    return scores
