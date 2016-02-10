from svgp_multi import SVGPMulti
from svgp_multi import SVGPMulti
from het_student_t import HetStudentT
import numpy as np
import GPy
X = np.random.randn(100,2)
Y = np.sin(X).sum(1)[:, None]
lik = HetStudentT()
kern = [GPy.kern.RBF(2, lengthscale=0.1) + GPy.kern.White(1, variance=1e-5), GPy.kern.RBF(2,lengthscale=1) + GPy.kern.White(2, variance=1e-5)]

mf = GPy.mappings.Constant(2,1, 3.0)

print X.shape
print Y.shape
m = SVGPMulti(X, Y, X[::2, :], kern, lik, mean_functions=[None, mf])

inf = GPy.inference.latent_function_inference.Laplace()
lik1 = GPy.likelihoods.StudentT()
kern1 = GPy.kern.RBF(1)
m1 = GPy.core.GP(X, Y, kernel=kern1, likelihood=lik1, inference_method=inf)

X = np.random.randn(100,2)
Y = np.sin(X).sum(1)[:, None]
lik = GPy.likelihoods.Gaussian()
kern = GPy.kern.RBF(2, lengthscale=0.1) + GPy.kern.White(1, variance=1e-5)

mf = GPy.mappings.Constant(2,1, 3.0)

print X.shape
print Y.shape
m2 = GPy.core.SVGP(X, Y, X[::2, :], kern, lik, mean_function=mf)
