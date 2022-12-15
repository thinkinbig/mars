#!/usr/bin/python
from cagd.vec import vec2

#solves the system of linear equations Ax = res
#where A is a tridiagonal matrix with diag2 representing the main diagonal
#diag1 and diag3 represent the lower and upper diagonal respectively
#all four parameters are vectors of size n
#the first element of diag1 and the last element of diag3 are ignored
#therefore diag1[i], diag2[i] and diag3[i] are located on the same row of A
def solve_tridiagonal_equation(diag1, diag2, diag3, res):
    # assert(len(diag1) == len(diag2) == len(diag3) == len(res))
    # solution = y = alpha = beta = nu =[0 for i in range(len(diag1))];
    # #initialize the first element
    # beta[0] = diag2[0];
    # nu[0] = diag3[0] / diag2[0];
    # #set up the value of three arrays according to the algorithm in assignment
    # for i in range(1,len(diag1)):
    #     alpha[i] = diag1[i];
    #     beta[i] = diag2[i] - alpha[i] * nu[i - 1];
    #     nu[i] = diag3[i] / beta[i];
    # #initialize first element of y
    # y[0] = res[0] / beta[0];
    # for i in range(len(1, diag1)):
    #     y[i] = (res[i] - alpha[i] * y[i - 1]) / beta[i]
    # #initialize first element of x
    # solution[len(diag1) - 1] = y[len(diag1) - 1]
    # for i in range(len(diag1) - 2, -1, -1):
    #     solution[i] = y[i] - nu[i] * solution[i + 1];
    # return solution

    assert(len(diag1) == len(diag2) == len(diag3) == len(res))
    # tridiagonal matrix algorithm
    n = len(diag1)
    solution = [res[0] * 0] * n
    # copy the original matrix
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

#solves the system of linear equations Ax = res
#where A is an almost tridiagonal matrix with diag2 representing the main diagonal
#diag1 and diag3 represent the lower and upper diagonal respectively
#all four parameters are vectors of size n
#the first element of diag1 and the last element of diag3 represent the top right and bottom left elements of A
#diag1[i], diag2[i] and diag3[i] are located on the same row of A
def solve_almost_tridiagonal_equation(diag1, diag2, diag3, res):
    assert(len(diag2) == len(diag3) == len(res) == len(diag1))
    n = len(diag1)
    # zero vector
    # initialize the array
    v, s, z, t = [0] * n, [0] * n, [0] * n, [0] * n
    # unify the vector with result
    y, w, solution = [res[0] * 0] * n, [res[0] * 0] * n, [res[0] * 0] * n
    # initialize the first element
    v[0], s[0] = 0, 1
    # initialize the last element
    t[-1] = 1
    # copy the original matrix
    a = diag1[:]
    b = diag2[:]
    c = diag3[:]
    d = res[:]
    # set up the value of three arrays according to the algorithm in assignment
    for i in range(1, n):
        z[i] = 1 / (b[i] + a[i] * v[i - 1])
        v[i] = -c[i] * z[i]
        y[i] = (d[i] - a[i] * y[i - 1]) * z[i]
        s[i] = -s[i - 1] * a[i] * z[i]
    for i in range(n - 2, -1, -1):
        t[i] = v[i] * t[i + 1] + s[i]
        w[i] = v[i] * w[i + 1] + y[i]
    solution[-1] = (d[-1] - a[-1] * w[-2] - c[-1] * w[0]) / (b[-1] + a[-1] * t[-2] + c[-1] * t[0])
    for i in range(n - 2, -1, -1):
        solution[i] = t[i] * solution[-1] + w[i]
    return solution

