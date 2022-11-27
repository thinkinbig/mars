#!/usr/bin/python
from cagd.vec import vec2

#solves the system of linear equations Ax = res
#where A is a tridiagonal matrix with diag2 representing the main diagonal
#diag1 and diag3 represent the lower and upper diagonal respectively
#all four parameters are vectors of size n
#the first element of diag1 and the last element of diag3 are ignored
#therefore diag1[i], diag2[i] and diag3[i] are located on the same row of A
def solve_tridiagonal_equation(diag1, diag2, diag3, res):
    assert(len(diag1) == len(diag2) == len(diag3) == len(res))
     assert(len(diag1) == len(diag2) == len(diag3) == len(res))
    solution = y = alpha = beta = nu =[0 for i in range(len(diag1))];
    
    #initialize the first element
    beta[0] = diag2[0];
    nu[0] = diag3[0] / diag2[0];
    #set up the value of three arrays according to the algorithm in assignment
    for i in range(1,len(diag1)):
        alpha[i] = diag1[i];
        beta[i] = diag2[i] - alpha[i] * nu[i - 1];
        nu[i] = diag3[i] / beta[i];
    #initialize first element of y
    y[0] = res[0] / beta[0];
    for i in range(len(1, diag1)):
        y[i] = (res[i] - alpha[i] * y[i - 1]) / beta[i]
    #initialize first element of x
    solution[len(diag1) - 1] = y[len(diag1) - 1]
    for i in range(len(diag1) - 2, -1, -1):
    solution[i] = y[i] - nu[i] * solution[i + 1];
    return solution
     

#solves the system of linear equations Ax = res
#where A is an almost tridiagonal matrix with diag2 representing the main diagonal
#diag1 and diag3 represent the lower and upper diagonal respectively
#all four parameters are vectors of size n
#the first element of diag1 and the last element of diag3 represent the top right and bottom left elements of A
#diag1[i], diag2[i] and diag3[i] are located on the same row of A
def solve_almost_tridiagonal_equation(diag1, diag2, diag3, res):
    assert(len(diag1) == len(diag2) == len(diag3) == len(res))
    # tridiagonal matrix algorithm
    n = len(diag1)
    solution = [0] * n
    _diag1 = diag1[:]
    _diag2 = diag2[:]
    _diag3 = diag3[:]
    _res = res[:]
    _diag3[0] /= _diag2[0]
    _res[0] /= _diag2[0]
    for i in range(1, n):
        _diag3[i] = _diag3[i] / (_diag2[i] - _diag1[i] * _diag3[i - 1])
        _res[i] = (_res[i] - _diag1[i] * _res[i - 1]) / (_diag2[i] - _diag1[i] * _diag3[i - 1])
    solution[n - 1] = _res[n - 1]
    for i in range(n - 2, -1, -1):
        solution[i] = _res[i] - _diag3[i] * solution[i + 1]
    return solution
