import numpy as np
import matplotlib.pyplot as plt
import math

def infocontent(p):
    
    # Alter the equation below to provide the correct Shannon information 
    # content:

    return -np.log2(p)

def entropy(p):  
    # First make sure the array is now a numpy array
    if type(p) != np.array:
        p = np.array(p)

    # Should we check any potential error conditions on the input?
    if (abs(np.sum(p) - 1) > 0.00001):
        raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(p))
    
    # We need to take the expectation value over the Shannon info content at
    # p(x) for each outcome x:
    weightedShannonInfos = p*(infocontent(p))
    # nansum ignores the nans from calling infocontent(0), but we still get the warning if an entry in p is zero
    return np.nansum(weightedShannonInfos)
