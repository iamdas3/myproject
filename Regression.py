
# coding: utf-8

# In[68]:


import requests
import pandas as pd
import scipy
import numpy as np
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def mean(values):
    return sum(values)/len(values)
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i]-mean_x)*(y[i]-mean_y)
    return covar
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])
def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    #print(response.content)
    d = pd.read_csv(TRAIN_DATA_URL, header = None)
    d_T = d.T
    #d_T = d_T[:].values()
    d_T.drop(d_T.index[1])
    #print(d_T)
    '''x_a = [row[0] for row in d]
    y_a = [row[1] for row in d]
    x_s = np.array(x_a[1:])
    y_s = np.array(y_a[1:])'''
    x_1 = d_T[0][1:]
    y_1 = d_T[1][1:]
    x_min = x_1.min()
    x_max = x_1.max()
    y_min = y_1.min()
    y_max = y_1.max()
    x = np.array((x_1-x_min)/(x_max-x_min))
    y = np.array((y_1-y_min)/(y_max-y_min))
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean/variance(x, x_mean))
    b0 = y_mean - b1*x_mean
    print(b0, b1)
    return np.array(b0+b1*area)
    
    
    
    
    


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")

