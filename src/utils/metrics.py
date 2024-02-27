import pandas as pd
import numpy as np
    
def mape(predictions, actuals):
    """Mean absolute percentage error"""
    return ((predictions - actuals).abs() / actuals).mean()

def rmse(predictions, actuals):
    """Root mean squared error"""
    return ((predictions - actuals)**2.mean())**0.5

def mae(predictions, actuals):
    """Mean absolute error"""
    return (predictions - actuals).mean()

def r(prediction, actuals): 
    """r-squared error"""
    ssreg = numpy.sum((actuals - prediction)**2)
    sstot = numpy.sum((actuals - actuals.mean())**2)
    
    return 1 - ssreg / sstot