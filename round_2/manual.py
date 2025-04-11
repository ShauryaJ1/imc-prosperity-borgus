import numpy as np
from scipy.optimize import minimize

#Find minimums for both using scipy

def minimize_ev(multiplier_1,individuals_1, multiplier_2, individuals_2):
    def function(x):
        p1,p2 = x
        return (multiplier_1 * 10000)/(individuals_1 + 100*p1) + (multiplier_2 * 10000)/(individuals_2 + 100*p2) - 50000

    bounds = [(0,1),(0,1)]
    constraints = [{
    'type': 'ineq',
    'fun': lambda x: 1 - x[0] - x[1]
     }]
    x0 = [1,1]
    return minimize(fun=function,x0=x0,bounds=bounds,constraints=constraints)



result = minimize_ev(10,1,80,6)

print(result)