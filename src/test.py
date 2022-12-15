#!/usr/bin/python

import cagd.utils as utils

diag1 = [3, 1.0, 1.0, 1.0, 1.0]
diag2 = [-2, -2, -2, -2, -2]
diag3 = [1.0, 1.0, 1.0, 1.0, 3]
res = [-1, -1, -1, -1, -1]
x = [5]
x = utils.solve_almost_tridiagonal_equation(diag1, diag2, diag3, res)
print(x)