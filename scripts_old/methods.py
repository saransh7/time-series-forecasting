import random
import scipy
import numpy as np
from math import pi
import pandas as pd
from numpy import arctan
import matplotlib.pyplot as plt
from itertools import combinations
from pandas import DataFrame, Series
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# calculating cv2 and adi of each group(time_series)
def group(df_series):
    time_series = list(df_series)
    demands =[temp for temp in time_series if temp != 0]
    std = np.std(demands)
    mean = np.mean(demands)
    DI = []
    sum = 1
    for temp in time_series[1:]:
        if temp == 0:
            sum += 1
        else:
            DI.append(sum)
            sum = 1
    ADI = np.mean(DI)
    CV2 = (float(std)/float(mean))**2
    # ADI vs CV2 1.32 0.49 
    return  CV2, ADI

#return next output for exponential smoothing
def exponential_smoothing(panda_series, alpha_value = 0.15 ):
    output = sum([alpha_value*(1 - alpha_value)** i * x for i,x in 
                    enumerate(reversed(panda_series))])
    return output
# moving average
def ma(df_series, evaluation_number, ma_index = 3):
    predicted_value = []
    df_series = list(df_series)
    for temp in range(evaluation_number,0,-1):
        sum = 0
        for i in range(1,ma_index+1):
            sum = sum + df_series[-temp - i]
        sum = float(sum)/float(ma_index)
        predicted_value.append(sum)
    actual_value = (df_series[-evaluation_number:])
    return predicted_value

#inputs are cronston evaluation_number = test_number alpha_value should be in between 0.15 to 0.20
def cronston(df_series, evaluation_number ,alpha_value = 0.2):
    #print df_series
    S = [df_series[0]]
    I = [1]
    M = []
    Q = []
    q = 1
    for element in df_series[1:]:
        if element == 0:
            S.append(S[-1])
            I.append(I[-1])
            q = q + 1
        else:
            S_t = alpha_value*element + (1 - alpha_value)*S[-1]
            #print "S_t '{}' ".format(S_t)
            I_t = alpha_value*q + (1 - alpha_value)*(I[-1])
            #print "I_t '{}'".format(I_t)
            Q.append(q)
            S.append(S_t)
            I.append(I_t)
            q = 1

        M.append(S[-1]/I[-1])

    predicted_value = M[-evaluation_number-1:-1]
    return predicted_value

# smart willemain model based on 
def smart_willemain(df_series, evaluation_number, r = 3):
    predicted_value = []
    actual_value = []
    #r = lead time
    for k in range(evaluation_number,0,-1): 
        df_series_temp = df_series[:-k]
        arr = []
        for i in range(10000):
            temp = 0
            for j in range(r):
                temp = temp + random.choice(df_series_temp)
            arr.append(temp)
        predicted_value.append(Series(arr).quantile(.95))
        #average of three consecutive weeks
        if evaluation_number > r-1:  
            act = sum(df_series[-k:-k + r])
        else:
            act = sum(df_series[-k:])

        actual_value.append(act)   

    return  predicted_value
    #plt.hist(arr,30)
    #plt.show()

# TODO : use RFE for feature selection

def get_features(df_series):
    # featrue vector = [last_non_zero element , last_non_zero element occurance, last element, ]
    # add 1 to each element as well for better use of weights
    flag_non0 = 2 if df_series[0] == 0 else 1
    last_non_zero = df_series[0] if df_series[0] != 0 else 0.01
    features = []
    i = 1
    for element in df_series[1:]:
        features.append([last_non_zero, flag_non0, df_series[i-1]])
        flag_non0 = flag_non0 + 1 if element == 0 else 1
        last_non_zero = element if element != 0 else last_non_zero
        i = i + 1 
    return features    
    # TODO : increasing feature size

def get_non_zero(df_series, temp, non_zero):
    if temp == 0:
        return non_zero
    else:
        a =  [i for i in list(df_series[:-temp]) if i != 0]
        non_zero.append(a)
        temp = temp - 1
        return get_non_zero(df_series, temp, non_zero)

def MASE(predicted_value, actual_value):
    # mean average scaled error
    predicted_value = list(predicted_value)
    actual_value = list(actual_value)
    e = []
    for act, pre in zip(actual_value, predicted_value):
        e.append(abs(act-pre))
    temp = 0
    i = 0
    for act in actual_value[1:]:
        temp = temp + abs(act - actual_value[i])
        i = i + 1 
    s = ((1/float(len(predicted_value) - 1))*temp)
    if s == 0:
        return 'NaN'
    else:
        return float(sum(e))/float(s)

def MAAPE(predicted_value, actual_value):
    # mean arctangent absolute percentage error
    predicted_value = list(predicted_value)
    actual_value = list(actual_value)
    sum = 0
    for pre, act in zip(predicted_value, actual_value):
        if act != 0:
            sum = sum + arctan(abs(pre - act)/float(act))
        else:
            sum = sum + pi/2

    return sum/(len(predicted_value))    

# logistic probability plus exponential smoothing
def logistic_regression(df_series, test_number):
    logreg = LogisticRegression(max_iter = 10000, warm_start=True, verbose=1 ,solver='newton-cg' )
    features = get_features(df_series)

    y = [0 if i == 0 else 1 for i in df_series]
    X = features[:-test_number]
    y = [0 if temp == 0 else 1 for temp in list(df_series[1:-test_number])]

    logreg.fit(X, y)
    prob = logreg.predict_proba(features[-test_number:])
    #print logreg.predict(features[-test_number:])
    smoothed_prediction = [exponential_smoothing(temp) for temp in get_non_zero(df_series, test_number, [[]])[1:]]
    predicted_value = []
    probabilities = []
    for probabilty, value in zip(prob, smoothed_prediction):
        predicted_value.append(probabilty[1]*value)
        probabilities.append(probabilty[1])
        
    return predicted_value, probabilities

if __name__ == '__main__':
    print "This script provides different methods for prediction"
    print "methods imlemented are:"
    print ''' 
            1. exponential smoothing
            2. cronston  
            3. smart_willemain
            4. mlp
            5. logistic regression
            '''