# Adaptive Autoregressive Support Vector Machine(AASVM)
import numpy as np
from scipy.optimize import minimize

#time_series = [1,2,3,4,5,6,7,8,9,10]
time_series = [1,0,1,0,1,0,0,1,0,1,0]
def autoCorrelation(time_series,k):
    # k = autocorrelation coeffecient btw time series D 1 + k to D t -1 and D1 to D t - 1 - k
    denm = np.var(time_series) * len(time_series)
    Dbar = np.mean(time_series)
    numr = 0
    for i in range(len(time_series) - k):
        numr = numr + (time_series[i] - Dbar) * (time_series[i + k] - Dbar)
    r = numr/float(denm)
    return r

def Xt(time_series, xt_index):
    Xt = []
    for i in range(len(time_series)-1,-1,-1):
        xt = []
        for lag in xt_index:
            if i - lag < 0:
                break
            else:
                xt.append(time_series[i-lag])
        if i - lag < 0:
            break
        else:
            Xt.append(xt)
    return Xt
    
def discrete_optimization(D,F):
    # using mse ae
    mse = []
    ae = []
    for d,f in zip(D,F):
        mse.append((d-f)**2)
        ae.append(abs(d-f))
    mse = [se/len(mse) for se in mse]
    norm = lambda array : [(x - min(array))/(max(array) - min(array)) for x in array] if max(array) != min(array) else [0 for x in array]
    mse = np.mean(norm(mse))
    ae = norm(ae)
    print "***************final_error******************"
    print (sum(ae) + mse)/2
    return (sum(ae) + mse)/2

def rbf ( Xi, xt, gamma):
    return -gamma*np.linalg.norm(np.array(Xi)-np.array(xt))

iter = 0

def fun(params, *args):
    rho, gamma = params
    D, k= args
    print rho
    rho = int(rho)
    xt_index = np.sort(np.argpartition(np.array(k), -rho)[-rho:])
    xt_index = [int(i) for i in list(xt_index + np.ones(len(xt_index)))]
    Xi = Xt(D, xt_index) 
    xt, Xi = Xi[0], Xi[1:]
    [a,a_prime] = np.split(np.array(sub_subproblem(time_series, gamma, Xi)), 2)
    #print a

    F = []
    for a, a_prime, Xi in zip(a, a_prime , Xi):
        F.append(((a-a_prime)) * rbf(Xi, xt, gamma))

    global iter
    iter = iter + 1
    print iter 
    
    D = list(reversed(D))
    return discrete_optimization(D,F)

def main_problem(time_series):
    # discrete optimization
    # minimizing nse, absolute and potential loss
    #time_series = list(reversed(time_series))
    ks = []
    K = len(time_series)
    for rho in range(1,K-1):
        ks.append(autoCorrelation(time_series[:-1], rho))
    
    cons = ({'type':'eq', 'fun' : lambda x : x[0]%1})
    result = minimize(fun, x0 = (2,1), constraints= cons , bounds = ((1,7), (0, None)) ,args = (time_series, ks))
    
    if result.success:
        fitted_params = result.x
        print(fitted_params)
    else:
        raise ValueError(result.message)

def convex_error(params, *args):
    params = np.split(np.array(params),2)
    [a, a_prime] = params
    D, gamma, Xi = args
    e = np.mean(D)/10
    error1 = (sum(a) + sum(a_prime))*e   
    error2  = 0
    error3  = 0
    for i in range(len(Xi)):
        error3 = error3 - 1*D[i]*(a[i] - a_prime[i]) 
        for j in range(len(Xi)):
            error2 = error2 + (a[i] - a_prime[i])*(a[j] - a_prime[j]) * rbf(Xi[i], Xi[j], gamma)
    print "-------------------sub error-------------------"
    print error1 + error2 + error3
    return error1 + error2 + error3

def fun2(x):
    #print x
    [a, a_prime] = np.split(x,2)
    return sum(a_prime * -1 + a)

def sub_subproblem(D, gamma, Xi):
    a = list(np.ones(len(Xi)))
    a_prime = list(np.zeros(len(Xi)))
    D = list(reversed(D[:-1]))
    C = 20
    cons = ({'type': 'eq', 'fun': fun2})
    result = minimize(convex_error ,x0 = (a + a_prime), constraints = cons, bounds= (((0,C),)*(2*len(a))) ,args = (D, gamma, Xi) )
    return result.x

if __name__ == '__main__':
    main_problem(time_series)