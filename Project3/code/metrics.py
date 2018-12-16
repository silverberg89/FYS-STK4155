# Required packages
import numpy as np

def mean_squared_error(y, yhat):
    
    """
    y: True values.
    yhat: Predictions.
    """
    
    res = y - yhat
    mse = np.divide(res.T@res, len(yhat))
    return mse

def r2_score(y, yhat):
    
    """
    y: True values.
    yhat: Predictions.
    """
    
    res = y - yhat
    ymean = np.mean(y)
    ssr = y - ymean * np.ones((len(y),))
    R2 = 1 - np.divide(res.T@res, ssr.T@ssr)
    return R2

def bias2(y, yhat):
    
    """
    y: True values.
    yhat: Predictions.
    """
    
    n = len(yhat)
    bias2 = np.sum((y - (np.mean(yhat)))**2) / n
    return bias2

def variance_error(yhat):
    
    """
    yhat: Predictions.
    """
    
    variance = np.mean(yhat**2) - np.mean(yhat)**2
    return variance 

def accuracy(y, yhat):
    
    """
    Metrics for binary data.
    y: True values.
    yhat: Predictions.
    """
    
    n = len(y)
    accuracy = np.sum(y == yhat) / n
    return accuracy 
    

