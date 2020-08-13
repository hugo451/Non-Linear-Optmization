import non_linear_optmization
from non_linear_optmization import NonLinearOptimization as NLOP
import numpy as np
import numdifftools as nd


f = lambda x: x[0]**2 + x[1]**2 + 1
z = np.array([2, 3])
n = 0.001
gradf = nd.Gradient(f)
d = -1 * gradf(z)

model = NLOP(z, f)

print(model.basic_algorithm())
print(model.gradient_descent())
print(model.newtons_metod())
print(model.conjugated_gradient())