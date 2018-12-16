# Required packages
import numpy as np

def to_binary(y, threshold):
    
    """
    Input: 
        y - vector
        threshold - a number between 0 and 1.
    Return: 
        A binary vector.   
    """
    
    binary = np.where(y > threshold, 1, 0)
    return binary