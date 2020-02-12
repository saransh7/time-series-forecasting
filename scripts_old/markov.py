import os
import numpy as np
from pandas import Series
from itertools import groupby
from cvxopt import matrix, solvers

t = [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0]
t2 = [0,0,1,1,0,2,1,0,1,2,0,1,2,0,1,2,0,1,0,1,0]
example = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,1,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,0,1,1,0,1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0]

def prob_matrix(M):
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [float(f)/float(s) for f in row]
    return M # prob matrix

def transit_matrix(transitions, step):
    n = 2 #number of states either 1 or zero
    M = [[0]*n for _ in range(n)]
    for(i,j) in zip(transitions, transitions[step:]):
        M[i][j] += 1
    return prob_matrix(M) #returning the transition prob matrix

def markov(df_series, k, test_number):
    #k is the number of lambdas of the transitions considered to predict the next value
    time_series = list(df_series)
    predicted_states = []
    transit_matrices = []
    for i in range(test_number,0,-1):
        train = time_series[:-i]
        X_hat = [len(list(group)) for key, group in groupby(sorted(train))]
        X_hat = [temp/float(sum(X_hat)) for temp in X_hat]
        
        QX = []
        temp_QX = []
        for q in range(k):
            Q = np.transpose(transit_matrix(train,q+1))
            transit_matrices.append(Q)
            QiXi = np.matmul(Q,X_hat)
            QX.append(QiXi)
            temp_QX.append(QiXi)
        estimates_no = len(QX)
        
        #modifying X_hat
        X_hat = np.array(X_hat)
        X_hat = np.append(-1*X_hat, X_hat)
        X_hat = np.append(X_hat, np.zeros(k+1))
        X_hat = np.append(X_hat, np.array([1,-1]))
        
        #modifying QX
        QX.append(np.array([-1]*len(QX[0])))

        # w >_ 0 lambda >_ 0
        I1 = [[0]*(k+1) for _ in range(k+1)]
        for i in range(k+1):
            I1[i][i] = -1
    
        X = []
        i = 0
        for qx in QX:
            if np.all(qx != QX[-1]):
                X.append([-x for x in list(qx)] + list(qx) + I1[i] + [1,-1])
            else:
                X.append(list(qx) + list(qx) + I1[i] + [0,0])
            i += 1
    
        QX = X
        QX = matrix(QX)
        X_hat = matrix(X_hat)
        
        sol = solvers.lp(matrix([0.]*(estimates_no) + [1]), QX, X_hat)
        estimates = np.round(sol['x'],4)
        sol = np.zeros(len(X_hat))
        estimates = list(np.reshape(estimates,estimates.shape[0]))
        estimates = estimates[:-1]
        predicted_state = np.zeros(2)
        #print list(reversed(train))
        for estimate, Q, state in zip(estimates, transit_matrices, reversed(train)):
            temp = np.zeros(2)
            temp[state] = 1
            #print temp
            predicted_state += np.matmul((estimate*Q), temp)
            
        predicted_states.append(np.round(predicted_state, 2))
    
    return predicted_states

def markov_prob(df_series, k , test_number):
    pred_prob = [x[1] for x in markov(df_series, k, test_number)]
    print len(pred_prob)
    return pred_prob

if __name__ == '__main__':

#  print markov(t,8,3)

  predictions = markov(t2,4,10)
  act = t2[-10:]
  for a,b in zip (act, predictions):
      print (a,b)