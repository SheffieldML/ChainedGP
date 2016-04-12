import jug
from jug import TaskGenerator, bvalue
from het_student_t import HetStudentT
from hetgp import HetGP
import pods
import GPy
from scipy.cluster.vq import kmeans as scipy_kmeans
import numpy as np
from svgp_multi import SVGPMulti
import pandas as pd
import scipy as sp
from sklearn import cross_validation
import copy
import os

from loglogistic import LogLogistic
from het_loglogistic import HetLogLogistic

Ms = [100]
#These are just starting values, lengthscales will also be randomized
f_rbf_lens = [0.4]
f_rbf_vars = [0.5]
g_rbf_lens = [0.5]
g_rbf_var = 0.5
gauss_noise = 0.25
# This is the log of mean of the posterior of the scale parameter, so we set it
# to be the log of roughly what we expect the scale parameter to be if it were
# constant
g_means = [np.log(gauss_noise)]
g_bias_vars = [None]
f_bias_vars = [None]#['mean']
fix_Zs = [False]#, True]

gauss_dataset_names = ['elevators1000', 'elevators10000']
stut_dataset_names = ['boston', 'motorCorrupt']
survival_dataset_names = ['simulated_survival', 'leukemia']
starting_df = 4.0
starting_r = 1.0

script_dir = os.path.dirname(__file__)
# mcycle_relative = 'data/mcycle.csv'
# elevator_relative = 'data/elevators.data'
# mcycle_path = os.path.join(script_dir, mcycle_relative)
# elevator_path = os.path.join(script_dir, elevator_relative)

optimize_dfs = [True]

full = True
if full:
    restarts = 5
    n_folds = 5
    preopt_scg_iters = 25
    preopt_restarts = 3
    opt_restarts = 3
    scg_iters = 25
    max_iters = 1500
    num_samples = 10000
    gtol = 1e-5
    ftol = 0
    xtol = 0
else:
    restarts = 1
    n_folds = 5
    preopt_scg_iters = 2
    preopt_restarts = 2
    opt_restarts = 2
    scg_iters = 3
    max_iters = 3
    num_samples = 1000
    gtol = 1e-5
    ftol = 0
    xtol = 0

folds = range(n_folds)
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class Dataset(object):
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest, Y_metadata_train=None, Y_metadata_test=None):
        #Hack to avoid survival model dieing
        if Y_metadata_train is None:
            Y_metadata_train = {'censored': np.zeros_like(Ytrain)}
            Ytrain = np.abs(Ytrain) + 0.01
        if Y_metadata_test is None:
            Y_metadata_test = {'censored': np.zeros_like(Ytest)}
            Ytest = np.abs(Ytest) + 0.01
        self.Xtrain, self.Ytrain, self.Xtest, self.Ytest, self.Y_metadata_train, self.Y_metadata_test = Xtrain, Ytrain, Xtest, Ytest, Y_metadata_train, Y_metadata_test

def log_pred_density(m, dataset, seed):
    return m.log_predictive_density_sampling(dataset.Xtest, dataset.Ytest, num_samples=num_samples, Y_metadata=dataset.Y_metadata_test)

def MAE(model, dataset, seed):
    mu, _ = model._raw_predict(dataset.Xtest)
    mu = model.likelihood.gp_link.transf(mu)
    return np.mean(np.abs(mu-dataset.Ytest))

def RMSE(model, dataset, seed):
    mu, _ = model._raw_predict(dataset.Xtest)
    mu = model.likelihood.gp_link.transf(mu)
    return np.sqrt(np.mean((mu-dataset.Ytest)**2))

def NMSE(model, dataset, seed):
    mu, _ = model._raw_predict(dataset.Xtest)
    mu = model.likelihood.gp_link.transf(mu)
    return np.sum((mu - dataset.Ytest)**2) / np.sum(dataset.Ytest - dataset.Ytrain.mean())

def random_multi_lengthscales_motor(X_):
    normed_X = (X_.max(0) - X_.min(0))/X_.std(0)
    #3 fluctuations in approximately the entire range
    f_lengthscales = np.random.uniform(size=X_.shape[1])*(0.3/normed_X) + 0.001
    g_lengthscales = np.random.uniform(size=X_.shape[1])*(0.3/normed_X) + 0.001
    #f_lengthscales = X_.std(0)*np.random.uniform(size=X_train.shape[1])*0.05 + 0.001
    #g_lengthscales = X_.std(0)*np.random.uniform(size=X_train.shape[1])*0.05 + 0.001
    #f_lengthscales = np.abs(-0.5 + np.random.uniform(size=X_train.shape[1])*2.5)
    #g_lengthscales = np.abs(-0.5 + np.random.uniform(size=X_train.shape[1])*2.5)
    return f_lengthscales, g_lengthscales

def random_multi_lengthscales_elevator(X_):
    normed_X = (X_.max(0) - X_.min(0))/X_.std(0)
    #3 fluctuations in approximately the entire range
    f_lengthscales = 0.2+np.random.uniform(size=X_.shape[1])*(0.4*normed_X) + 0.001
    g_lengthscales = 0.2+np.random.uniform(size=X_.shape[1])*(0.4*normed_X) + 0.001
    return f_lengthscales, g_lengthscales

def random_multi_lengthscales_boston(X_):
    f_lengthscales = 0.5 + np.random.uniform(size=X_.shape[1])*2.5 + 0.001
    g_lengthscales = 0.5 + np.random.uniform(size=X_.shape[1])*2.5 + 0.001
    return f_lengthscales, g_lengthscales

def random_multi_lengthscales_leukemia(X_):
    f_lengthscales = 0.5 + np.random.uniform(size=X_.shape[1])*2.5 + 0.001
    g_lengthscales = 0.5 + np.random.uniform(size=X_.shape[1])*2.5 + 0.001
    return f_lengthscales, g_lengthscales

def random_multi_lengthscales_simulated_survival(X_):
    f_lengthscales = 0.5 + np.random.uniform(size=X_.shape[1])*0.5 + 0.001
    g_lengthscales = 0.5 + np.random.uniform(size=X_.shape[1])*0.5 + 0.001
    return f_lengthscales, g_lengthscales


def build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    kernf = GPy.kern.RBF(D, variance=f_rbf_var,
                         lengthscale=np.ones(D)*f_rbf_len, ARD=True,
                         name='kernf_rbf')
    #kernf += GPy.kern.Linear(D, ARD=True, name='kernf_linear')
    kernf += GPy.kern.White(1, variance=0.001, name='f_white')
    if f_bias_var is not None:
        if f_bias_var == 'mean':
            f_bias_var = dataset.Ytrain.mean()
        kernf += GPy.kern.Bias(1, variance=f_bias_var, name='f_bias')
    kernf.f_white.fix()
    kernf.name = 'kernf'
    return kernf

def build_kerng(D, g_bias_var, g_rbf_len, seed, fold):
    #Needs white or variance doesn't checkgrad!
    kerng = GPy.kern.RBF(D, variance=g_rbf_var,
                         lengthscale=np.ones(D)*g_rbf_len, ARD=True,
                         name='kerng_rbf')
    kerng += GPy.kern.White(1, variance=0.001, name='g_white')
    if g_bias_var is not None:
        kerng += GPy.kern.Bias(1, variance=g_bias_var, name='g_bias')
    kerng.g_white.fix()
    kerng.name = 'kerng'
    return kerng

def kmeans(dataset, k, seed):
    Z, _ = scipy_kmeans(dataset.Xtrain, k)
    return Z

def remove_non_censored(dataset):
    dataset_copy = copy.deepcopy(dataset)
    if dataset_copy.Y_metadata_train is not None:
        non_censored_inds = dataset_copy.Y_metadata_train['censored'][:, 0] < 0.5
    dataset_copy.Xtrain = dataset_copy.Xtrain[non_censored_inds, :]
    dataset_copy.Ytrain = dataset_copy.Ytrain[non_censored_inds, :]
    if dataset_copy.Y_metadata_test is not None:
        non_censored_inds = dataset_copy.Y_metadata_test['censored'][:, 0] < 0.5
    dataset_copy.Xtest = dataset_copy.Xtest[non_censored_inds, :]
    dataset_copy.Ytest = dataset_copy.Ytest[non_censored_inds, :]
    return dataset_copy

def remove_huge(dataset):
    if dataset.Xtrain.shape[0] > 2000:
        dataset_copy = copy.deepcopy(dataset)
        dataset_copy.Xtrain = dataset_copy.Xtrain[:5, :]
        dataset_copy.Ytrain = dataset_copy.Ytrain[:5, :]
        if dataset_copy.Y_metadata_train is not None:
            dataset_copy.Y_metadata_train = {'censored': dataset_copy.Y_metadata_train['censored'][:5, :]}
        return dataset_copy
    else:
        return dataset

def build_gauss_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    dataset = remove_non_censored(dataset)
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    #gauss likelihood baseline
    #likelihood = GPy.likelihoods.gauss()
    #m_gauss = GPy.core.SVGP(dataset.Xtrain, dataset.Ytrain, Z=Z.copy(), kernel=kernf,
    #                        likelihood=likelihood, name='gauss_single')
    m_gauss = GPy.models.SparseGPRegression(dataset.Xtrain.copy(), dataset.Ytrain.copy(), Z=Z.copy(), kernel=kernf)
    m_gauss.name='gauss_single'
    m_gauss.likelihood.variance[:] = gauss_noise
    return m_gauss

def build_multi_gauss_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold):
    dataset = remove_non_censored(dataset)
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    kerng = build_kerng(D, g_bias_var, g_rbf_len, seed, fold)
    kern = [kernf, kerng]

    #Multiple latent process model
    if g_mean is not None:
        g_mean = GPy.mappings.Constant(input_dim=13, output_dim=1, value=g_mean)
    mean_functions = [None, g_mean]
    likelihood = HetGP()
    m_multi = SVGPMulti(dataset.Xtrain.copy(), dataset.Ytrain.copy(), Z.copy(), kern_list=kern,
                        likelihood=likelihood, mean_functions=mean_functions, name='multi_gauss')
    return m_multi

def build_stut_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    dataset = remove_non_censored(dataset)
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)

    #Single latent process model
    stu_t = GPy.likelihoods.StudentT(deg_free=starting_df)
    m_stut = GPy.core.SVGP(dataset.Xtrain.copy(), dataset.Ytrain.copy(), Z=Z.copy(), kernel=kernf,
                             likelihood=stu_t, name='student_t_single')
    stu_t.t_scale2[:] = gauss_noise
    return m_stut

def build_laplace_stut_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    dataset = remove_non_censored(dataset)
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)

    #Single latent process model
    stu_t = GPy.likelihoods.StudentT(deg_free=starting_df)
    m_stut = GPy.core.GP(X=dataset.Xtrain.copy(), Y=dataset.Ytrain.copy(), kernel=kernf,
                             likelihood=stu_t, inference_method=GPy.inference.latent_function_inference.Laplace(), name='laplace_student_t_single')
    stu_t.t_scale2[:] = gauss_noise
    return m_stut


def build_multi_stut_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold):
    dataset = remove_non_censored(dataset)
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    kerng = build_kerng(D, g_bias_var, g_rbf_len, seed, fold)
    kern = [kernf, kerng]

    #Multiple latent process model
    if g_mean is not None:
        g_mean = GPy.mappings.Constant(input_dim=dataset.Xtrain.shape[1], output_dim=1, value=g_mean)
    mean_functions = [None, g_mean]
    likelihood = HetStudentT(deg_free=starting_df)
    m_multi = SVGPMulti(dataset.Xtrain.copy(), dataset.Ytrain.copy(), Z.copy(), kern_list=kern,
                        likelihood=likelihood, mean_functions=mean_functions, name='student_t_multi')
    return m_multi

def build_survival_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)

    #Single latent process model
    loglogistic = LogLogistic(r=starting_r)
    m_loglogistic = GPy.core.SVGP(dataset.Xtrain.copy(), dataset.Ytrain.copy(), Z=Z.copy(), kernel=kernf,
                             likelihood=loglogistic, name='survival_t_single', Y_metadata=dataset.Y_metadata_train)
    return m_loglogistic

def build_laplace_survival_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)

    #Single latent process model
    loglogistic = LogLogistic(r=starting_r)
    m_loglogistic = GPy.core.GP(X=dataset.Xtrain.copy(), Y=dataset.Ytrain.copy(), kernel=kernf,
                                likelihood=loglogistic, inference_method=GPy.inference.latent_function_inference.Laplace(),
                                name='laplace_survival_t_single', Y_metadata=dataset.Y_metadata_train)
    return m_loglogistic


def build_multi_survival_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold):
    D = dataset.Xtrain.shape[1]
    kernf = build_kernf(D, dataset, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    kerng = build_kerng(D, g_bias_var, g_rbf_len, seed, fold)
    kern = [kernf, kerng]

    #Multiple latent process model
    if g_mean is not None:
        g_mean = GPy.mappings.Constant(input_dim=dataset.Xtrain.shape[1], output_dim=1, value=g_mean)
    mean_functions = [None, g_mean]
    likelihood = HetLogLogistic()
    m_multi = SVGPMulti(dataset.Xtrain.copy(), dataset.Ytrain.copy(), Z.copy(), kern_list=kern,
                        likelihood=likelihood, mean_functions=mean_functions, name='survival_t_multi', Y_metadata=dataset.Y_metadata_train)
    return m_multi


"""
Methods for optimizing models
"""
def preopt_gauss_scheme(m):
    if hasattr(m, 'Z'):
        m.Z.fix()
    # m.kernf.constrain_positive()
    m.kernf.f_white.fix()
    m.likelihood.variance.constrain_positive()
    m.kernf.fix()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        m.optimize('scg', max_iters=2*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())
    m.kernf.constrain_positive()
    m.kernf.f_white.fix()
    m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    return m

def preopt_stut_scheme(m):
    if hasattr(m, 'Z'):
        m.Z.fix()
    m.kernf.fix()
    m.kernf.f_white.fix()
    m.likelihood.t_scale2.constrain_positive()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=2*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"

        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())
    m.kernf.constrain_positive()
    m.kernf.f_white.fix()
    try:
        m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        pass
    return m

def preopt_laplace_stut_scheme(m):
    if hasattr(m, 'Z'):
        m.Z.fix()
    m.kernf.constrain_positive()
    m.kernf.f_white.fix()
    m.likelihood.t_scale2.constrain_positive()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=2*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())

    try:
        m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        pass
    return m

def preopt_survival_scheme(m):
    if hasattr(m, 'Z'):
        m.Z.fix()
    m.kernf.fix()
    m.kernf.f_white.fix()
    m.likelihood.r_shape.constrain_positive()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=2*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())
    m.kernf.constrain_positive()
    m.kernf.f_white.fix()
    try:
        m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        pass

    return m

def preopt_laplace_survival_scheme(m):
    if hasattr(m, 'Z'):
        m.Z.fix()
    m.kernf.constrain_positive()
    m.kernf.f_white.fix()
    m.likelihood.r_shape.constrain_positive()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=2*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())
    m.kernf.f_white.fix()
    try:
        m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        pass
    return m

def preopt_multi_gauss_scheme(m):
    #We wish to optimize the model for a little while with things fixed, then unfix
    m.kernf.fix()
    m.kerng.fix()
    if hasattr(m, 'Z'):
        m.Z.fix()
    if hasattr(m, 'constmap'):
        m.constmap.fix()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=1*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())
    #Constrain all kernel parameters positive and reoptimize
    m.kernf.constrain_positive()
    m.kerng.constrain_positive()
    m.kernf.f_white.fix()
    m.kerng.g_white.fix()
    if hasattr(m, 'constmap'):
        m.constmap.unfix()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=1*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood continueing", np.sum(m.log_likelihood())
            break
    try:
        m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        pass
    return m

def preopt_multi_stut_scheme(m):
    #We wish to optimize the model for a little while with things fixed, then unfix
    m.kernf.fix()
    m.kerng.fix()
    if hasattr(m, 'Z'):
        m.Z.fix()
    if hasattr(m, 'constmap'):
        m.constmap.fix()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=1*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"

        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())
    #Constrain all kernel parameters positive and reoptimize
    m.kernf.constrain_positive()
    m.kerng.constrain_positive()
    m.kernf.f_white.fix()
    m.kerng.g_white.fix()
    if hasattr(m, 'constmap'):
        m.constmap.unfix()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=1*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood continueing", np.sum(m.log_likelihood())
            break
    try:
        m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        pass
    return m

def preopt_multi_survival_scheme(m):
    #We wish to optimize the model for a little while with things fixed, then unfix
    m.kernf.fix()
    m.kerng.fix()
    if hasattr(m, 'Z'):
        m.Z.fix()
    if hasattr(m, 'constmap'):
        m.constmap.fix()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=1*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"

        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood ", np.sum(m.log_likelihood())
            break
        else:
            old_ll = np.sum(m.log_likelihood())
    #Constrain all kernel parameters positive and reoptimize
    m.kernf.constrain_positive()
    m.kerng.constrain_positive()
    m.kernf.f_white.fix()
    m.kerng.g_white.fix()
    if hasattr(m, 'constmap'):
        m.constmap.unfix()
    for i in range(preopt_restarts):
        old_ll = np.sum(m.log_likelihood())
        try:
            m.optimize('scg', max_iters=1*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        except:
            print "preopt failed"
        if np.allclose(old_ll, np.sum(m.log_likelihood())):
            print "Stuck with log likelihood continueing", np.sum(m.log_likelihood())
            break
    try:
        m.optimize('bfgs', max_iters=5*preopt_scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        pass
    return m


"""
Methods for finding a good model to do further optimization with
"""

@TaskGenerator
def preopt_gauss(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, random_func, seed, fold):
    m = build_gauss_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    print "PreOptimizing gauss ", m
    print m.kernf.kernf_rbf.lengthscale
    best_m = preopt_gauss_scheme(m)
    for i in range(restarts):
        #Randomize restarts
        lens, _ = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m.likelihood.variance[:] = gauss_noise
        m = preopt_gauss_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+5: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best gauss model"
    print best_m
    return best_m[:]

@TaskGenerator
def preopt_multi_gauss(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, random_func, seed, fold):
    m = build_multi_gauss_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m_qu_means = m.q_u_means.values.copy()
    m_qu_chols = m.q_u_chols.values.copy()
    print "PreOptimizing multi gauss ", m
    print m.kernf.kernf_rbf.lengthscale
    print m.kerng.kerng_rbf.lengthscale
    best_m = preopt_multi_gauss_scheme(m)

    for i in range(restarts):
        #Randomize restarts
        f_lens, g_lens = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = f_lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m.kerng.kerng_rbf.lengthscale[:] = g_lens
        m.kerng.kerng_rbf.variance[:] = g_rbf_var
        if g_mean is not None:
            m.constmap[:] = g_mean
        m.q_u_means[:] = m_qu_means
        m.q_u_chols[:] = m_qu_chols
        m = preopt_multi_gauss_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+5: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best multi gauss model"
    print best_m
    return best_m[:]

@TaskGenerator
def preopt_stut(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, random_func, seed, fold):
    m = build_stut_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    print "PreOptimizing stut", m
    print m.kernf.kernf_rbf.lengthscale
    best_m = preopt_stut_scheme(m)
    for i in range(restarts):
        #Randomize restarts
        lens, _ = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m.likelihood.t_scale2[:] = gauss_noise
        m = preopt_stut_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+5: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best stut tmodel"
    print best_m
    return best_m[:]

@TaskGenerator
def preopt_laplace_stut(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, random_func, seed, fold):
    m = build_laplace_stut_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    print "PreOptimizing stut", m
    print m.kernf.kernf_rbf.lengthscale
    best_m = preopt_stut_scheme(m)
    for i in range(restarts):
        #Randomize restarts
        lens, _ = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m.likelihood.t_scale2[:] = gauss_noise
        m = preopt_stut_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+5: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best stut tmodel"
    print best_m
    return best_m[:]

@TaskGenerator
def preopt_stut_multi(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, random_func, seed, fold):
    m = build_multi_stut_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m_qu_means = m.q_u_means.values.copy()
    m_qu_chols = m.q_u_chols.values.copy()
    print "PreOptimizing multi stut", m
    print m.kernf.kernf_rbf.lengthscale
    print m.kerng.kerng_rbf.lengthscale
    best_m = preopt_multi_stut_scheme(m)
    for i in range(restarts):
        #Randomize restarts
        f_lens, g_lens = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = f_lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m.kerng.kerng_rbf.lengthscale[:] = g_lens
        m.kerng.kerng_rbf.variance[:] = g_rbf_var
        if g_mean is not None:
            m.constmap[:] = g_mean
        m.q_u_means[:] = m_qu_means
        m.q_u_chols[:] = m_qu_chols
        m = preopt_multi_stut_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+8: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best multi stut model"
    print best_m
    return best_m[:]

@TaskGenerator
def preopt_survival(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, random_func, seed, fold):
    m = build_survival_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    print "PreOptimizing survival", m
    print m.kernf.kernf_rbf.lengthscale
    best_m = preopt_survival_scheme(m)
    for i in range(restarts):
        #Randomize restarts
        lens, _ = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m = preopt_survival_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+8: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best survival tmodel"
    print best_m
    return best_m[:]

@TaskGenerator
def preopt_laplace_survival(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, random_func, seed, fold):
    m = build_laplace_survival_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    print "PreOptimizing survival", m
    print m.kernf.kernf_rbf.lengthscale
    best_m = preopt_survival_scheme(m)
    for i in range(restarts):
        #Randomize restarts
        lens, _ = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m = preopt_survival_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+8: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best survival tmodel"
    print best_m
    return best_m[:]


@TaskGenerator
def preopt_survival_multi(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, random_func, seed, fold):
    m = build_multi_survival_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m_qu_means = m.q_u_means.values.copy()
    m_qu_chols = m.q_u_chols.values.copy()
    print "PreOptimizing multi survival", m
    print m.kernf.kernf_rbf.lengthscale
    print m.kerng.kerng_rbf.lengthscale
    best_m = preopt_multi_survival_scheme(m)
    for i in range(restarts):
        #Randomize restarts
        f_lens, g_lens = random_func(m.X.values)
        m.kernf.kernf_rbf.lengthscale[:] = f_lens
        m.kernf.kernf_rbf.variance[:] = f_rbf_var
        m.kerng.kerng_rbf.lengthscale[:] = g_lens
        m.kerng.kerng_rbf.variance[:] = g_rbf_var
        if g_mean is not None:
            m.constmap[:] = g_mean
        m.q_u_means[:] = m_qu_means
        m.q_u_chols[:] = m_qu_chols
        m = preopt_multi_survival_scheme(m)
        if np.sum(m.log_likelihood()) > np.sum(best_m.log_likelihood()):
            if np.sum(m.log_likelihood()) < 1e+8: #This is to avoid exploding models being selected
                best_m[:] = m[:].copy()
    print "Found best multi survival model"
    print best_m
    return best_m[:]


"""
Methods to fully optimize models
"""
@TaskGenerator
def optimize_gauss(m_param, dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    m = build_gauss_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        print "Model broke down"
    print "Learnt model: "
    print m
    return m[:]

@TaskGenerator
def optimize_multi_gauss(m_param, dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold):
    m = build_multi_gauss_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
    except:
        print "Model broken down"
    print "Learnt model: "
    print m
    return m[:]

@TaskGenerator
def optimize_stut(m_param, dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, optimize_df, seed, fold):
    m = build_stut_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        if optimize_df:
            m.likelihood.deg_free.constrain_positive()
        m.optimize('scg', max_iters=int(max_iters/2), messages=1)
    except:
        print "Model broken down"
    print "Learnt model: "
    print m
    return m[:]

@TaskGenerator
def optimize_laplace_stut(m_param, dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, optimize_df, seed, fold):
    m = build_laplace_stut_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        if optimize_df:
            m.likelihood.deg_free.constrain_positive()
        m.optimize('scg', max_iters=int(max_iters/2), messages=1)
    except:
        print "Model broken down"
    print "Learnt model: "
    print m
    return m[:]


@TaskGenerator
def optimize_multi_stut(m_param, dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, optimize_df, seed, fold, ):
    m = build_multi_stut_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        if optimize_df:
            m.likelihood.deg_free.constrain_positive()
        m.optimize('scg', max_iters=int(max_iters/2), messages=1)
    except:
        print "Model broken down"
    print "Learnt model: "
    print m
    return m[:]

@TaskGenerator
def optimize_survival(m_param, dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    m = build_survival_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        m.optimize('scg', max_iters=int(max_iters/2), messages=1)
    except:
        print "Model broken down"
    print "Learnt model: "
    print m
    return m[:]

@TaskGenerator
def optimize_laplace_survival(m_param, dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold):
    m = build_laplace_survival_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        m.optimize('scg', max_iters=int(max_iters/2), messages=1)
    except:
        print "Model broken down"
    print "Learnt model: "
    print m
    return m[:]


@TaskGenerator
def optimize_multi_survival(m_param, dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold, ):
    m = build_multi_survival_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m[:] = m_param[:]
    print "Optimizing gauss ", m
    print m.X.shape
    print m.Y.shape
    if hasattr(m, 'Z'):
        if not fixZ:
            m.Z.unfix()
        else:
            m.Z.fix()
    try:
        # Optimize Z for a bit then start the hard work
        [m.optimize('scg', max_iters=scg_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
         for _ in range(opt_restarts)]
        m.optimize('bfgs', max_iters=max_iters, gtol=gtol, messages=1, xtol=xtol, ftol=ftol)
        m.optimize('scg', max_iters=int(max_iters/2), messages=1)
    except:
        print "Model broken down"
    print "Learnt model: "
    print m
    return m[:]


"""
Evaluation function
"""
@TaskGenerator
def evaluate(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold,
             m_gauss_opt_params, m_multi_gauss_opt_params,
             m_stut_opt_params=None, m_laplace_stut_opt_params=None, m_multi_stut_opt_params=None,
             m_survival_opt_params=None, m_laplace_survival_opt_params=None, m_multi_survival_opt_params=None,
             student_t=False, survival=False):
    # If the dataset will cause memory issues for laplace, just replace with a toy dataset
    laplace_dataset = remove_huge(dataset)
    # Make the models to compare
    print "Finished optimizing, evaluation now"
    m_gauss_opt = build_gauss_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m_gauss_opt[:] = m_gauss_opt_params
    m_multi_gauss_opt = build_multi_gauss_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m_multi_gauss_opt[:] = m_multi_gauss_opt_params

    m_stut_opt = build_stut_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m_stut_opt[:] = m_stut_opt_params
    m_laplace_stut_opt = build_laplace_stut_model(laplace_dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m_laplace_stut_opt[:] = m_laplace_stut_opt_params
    m_multi_stut_opt = build_multi_stut_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m_multi_stut_opt[:] = m_multi_stut_opt_params

    m_survival_opt = build_survival_model(dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m_survival_opt[:] = m_survival_opt_params
    m_laplace_survival_opt = build_laplace_survival_model(laplace_dataset, Z, fixZ, f_bias_var, f_rbf_len, f_rbf_var, seed, fold)
    m_laplace_survival_opt[:] = m_laplace_survival_opt_params
    m_multi_survival_opt = build_multi_survival_model(dataset, Z, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, seed, fold)
    m_multi_survival_opt[:] = m_multi_survival_opt_params

    opt_list = [m_gauss_opt, m_multi_gauss_opt, m_stut_opt, m_laplace_stut_opt, m_multi_stut_opt, m_survival_opt, m_laplace_survival_opt, m_multi_survival_opt]

    # num_models = len(opt_list)
    # num_data = dataset.Ytest.shape[0]
    # num_dims = dataset.Ytest.shape[1]
    # log_preds = np.zeros((num_models, num_data))*np.nan
    # RMSEs = np.zeros(num_models)*np.nan
    # MAEs = np.zeros(num_models)*np.nan
    # NMSEs = np.zeros(num_models)*np.nan
    # log_likelihoods = np.zeros(num_models)*np.nan
    # predictions = np.zeros((num_models, 2, num_data, num_dims))*np.nan

    #Need to fix the below as they would have to hash a model = bad
    #Compute LPD
    # for i, m_opt in enumerate(opt_list):
        # log_preds[i, :] = log_pred_density(m_opt, dataset, seed)
        # RMSEs[i] = RMSE(m_opt, dataset, seed)
        # MAEs[i] = MAE(m_opt, dataset, seed)
        # NMSEs[i] = NMSE(m_opt, dataset, seed)
        # log_likelihoods[i] = np.sum(m_opt.log_likelihood())
        # predictions[i,:,:,:] = m_opt._raw_predict(dataset.Xtest)

    log_preds = [log_pred_density(m_opt, dataset, seed) for m_opt in opt_list]
    RMSEs = [RMSE(m_opt, dataset, seed) for m_opt in opt_list]
    MAEs = [MAE(m_opt, dataset, seed) for m_opt in opt_list]
    NMSEs = [NMSE(m_opt, dataset, seed) for m_opt in opt_list]
    log_likelihoods = [np.sum(m_opt.log_likelihood()) for m_opt in opt_list]
    predictions = [m_opt._raw_predict(dataset.Xtest) for m_opt in opt_list]

    log_preds = np.hstack(log_preds)
    RMSEs = np.hstack(RMSEs)
    MAEs = np.hstack(MAEs)
    log_likelihoods = np.array(log_likelihoods)
    predictions = np.array(predictions)
    NMSEs = np.array(NMSEs)
    results = np.array([log_preds, RMSEs, MAEs, log_likelihoods, predictions, NMSEs])
    print "Evaluated"
    return results

"""
Loading different datasets
"""
def load_elevators(fold, seed, num_training):
    np.random.seed(seed)
    data = pods.datasets.elevators(seed=seed)
    X = data['X']
    Y = data['Y']

    X = (X-X.mean(0))/X.std(0)
    Y = (Y-Y.mean(0))/Y.std(0)

    cv = cross_validation.KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
    #Stupidly you can't just index it...
    for i, (train, test) in enumerate(cv):
        if i==fold:
            training_inds = train
            test_inds = test
    if training_inds.shape[0] < num_training:
        raise ValueError()
    if test_inds.shape[0] < 1500:
        raise ValueError()
    training_inds = training_inds[:num_training]
    #Use the final 1500 so we test on the same points for both models
    test_inds = test_inds[-1500:]
    Xtrain = X[training_inds, :]
    Ytrain = Y[training_inds, :]
    Xtest = X[test_inds, :]
    Ytest = Y[test_inds, :]

    print "training shape: ", Xtrain.shape
    print "test shape: ", Xtest.shape
    print "All: ", X.shape
    print Xtrain.shape[0] + Xtest.shape[0] - X.shape[0]
    return Dataset(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest)

def load_yuan(fold, seed, num_training=200):
    np.random.seed(seed)
    X = np.random.uniform(size=num_training)[:, None]
    muX = 2*(np.exp(-30.0*(X - 0.25)**2) + np.sin(np.pi*X**2)) - 2
    gX = np.sin(2*np.pi*X)
    Y = np.random.multivariate_normal(mean=muX.flatten(), cov=np.diagflat(np.exp(gX)))[:, None]
    X = (X-X.mean(0))/X.std(0)
    Y = (Y-Y.mean(0))/Y.std(0)

    cv = cross_validation.KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
    #Stupidly you can't just index it...
    for i, (train, test) in enumerate(cv):
        if i==fold:
            training_inds = train
            test_inds = test

    Xtrain = X[training_inds, :]
    Ytrain = Y[training_inds, :]
    Xtest = X[test_inds, :]
    Ytest = Y[test_inds, :]

    print "training shape: ", Xtrain.shape
    print "test shape: ", Xtest.shape
    print "All: ", X.shape
    print Xtrain.shape[0] + Xtest.shape[0] - X.shape[0]
    return Dataset(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest)

def load_boston(fold, seed):
    data = pods.datasets.boston_housing()
    X = data['X']
    Y = data['Y']
    #N = Y.shape[0]
    X = (X-X.mean(0))/X.std(0)
    Y = (Y-Y.mean(0))/Y.std(0)

    #num_training = int(N*percent_train/100.0)
    #offset = fold*num_training
    #training_inds = np.random.permutation(range(Y.shape[0]))[offset:(offset+num_training)]
    #test_inds = np.setdiff1d(np.array(range(Y.shape[0])), training_inds)
    cv = cross_validation.KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
    #Stupidly you can't just index it...
    for i, (train, test) in enumerate(cv):
        if i==fold:
            training_inds = train
            test_inds = test
    Xtrain = X[training_inds, :]
    Ytrain = Y[training_inds, :]
    Xtest = X[test_inds, :]
    Ytest = Y[test_inds, :]

    print "training shape: ", Xtrain.shape
    print "test shape: ", Xtest.shape
    print "All: ", X.shape
    print Xtrain.shape[0] + Xtest.shape[0] - X.shape[0]
    return Dataset(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest)

def load_motorCorrupt(fold, seed):
    np.random.seed(seed)
    data = pods.datasets.mcycle(seed=seed)

    X = data['X']
    Y = data['Y']
    X = (X-X.mean(0))/X.std(0)
    Y = (Y-Y.mean(0))/Y.std(0)

    cv = cross_validation.KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
    #Stupidly you can't just index it...
    for i, (train, test) in enumerate(cv):
        if i==fold:
            training_inds = train
            test_inds = test

    #Corrupt some Y's with larger noise
    corrupt_inds = np.random.permutation(range(X.shape[0]))[:45]
    Y[corrupt_inds] = np.random.randn(*Y[corrupt_inds].shape)*4.0

    Xtrain = X[training_inds, :]
    Ytrain = Y[training_inds, :]
    Xtest = X[test_inds, :]
    Ytest = Y[test_inds, :]

    #Corrupt 20 training Y's with larger noise
    #corrupt_inds = np.random.permutation(range(Xtrain.shape[0]))[:10]
    #Ytrain[corrupt_inds] += np.random.randn(*Ytrain[corrupt_inds].shape)*2.0

    print "training shape: ", Xtrain.shape
    print "test shape: ", Xtest.shape
    print "All: ", X.shape
    print Xtrain.shape[0] + Xtest.shape[0] - X.shape[0]
    return Dataset(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest)

def load_leukemia(fold, seed):
    np.random.seed(seed)
    data = pods.datasets.leukemia()

    #IMPORTANT, 1 means censored!!!
    lc = 1-data['censoring'][:, None]
    S = data['Y'][:, None]
    Xcovs = data['X'][:, :]
    #Just 'age', 'sex', 'wbc', and 'tpi'
    Xcovs = Xcovs[:, 3:7]

    #Unskew the WBC
    Xcovs[:,2] = np.log10(Xcovs[:,2] + 0.3)
    #Normalise continuous values to mean 0, std 1
    cond_inds = np.array([0, 2, 3])
    Xcovs[:, cond_inds] = (Xcovs[:, cond_inds] - np.mean(Xcovs[:, cond_inds], axis=0))/np.std(Xcovs[:, cond_inds], axis=0)
    S_scale = float(sp.stats.gmean(S))
    S = S/S_scale

    cv = cross_validation.KFold(n=Xcovs.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
    #Stupidly you can't just index it...
    for i, (train, test) in enumerate(cv):
        if i==fold:
            training_inds = train
            test_inds = test

    Xtrain = Xcovs[training_inds, :]
    Ytrain = S[training_inds, :]
    Xtest = Xcovs[test_inds, :]
    Ytest = S[test_inds, :]
    Y_metadata_train = {'censored': lc[training_inds, :]}
    Y_metadata_test = {'censored': lc[test_inds, :]}

    print "training shape: ", Xtrain.shape
    print "test shape: ", Xtest.shape
    print "All: ", Xcovs.shape
    print Xtrain.shape[0] + Xtest.shape[0] - Xcovs.shape[0]
    return Dataset(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest,
                   Y_metadata_train=Y_metadata_train, Y_metadata_test=Y_metadata_test)

def load_simulated_survival(fold, seed):
    np.random.seed(seed)
    N = 1000
    cens_perc= 20.0
    D = 2

    if D == 1:
        X = np.linspace(0,1,N)[:, None]
    else:
        X = np.random.uniform(0, 1, size=(N,D))
    kern_f = GPy.kern.RBF(X.shape[1], lengthscale=0.2, name='f_rbf')
    kern_f += GPy.kern.White(X.shape[1], variance=1e-5, name='f_white')
    kern_g = GPy.kern.RBF(X.shape[1], lengthscale=0.3, name='g_rbf')
    #kern_g += GPy.kern.Bias(X.shape[1], variance=5.0, name='g_bias')
    kern_g += GPy.kern.White(X.shape[1], variance=1e-5, name='g_white')
    kern_g.name = 'g_kern'
    kern_f.name = 'f_kern'
    #kern_g = GPy.kern.Bias(X.shape[1])
    Kf = kern_f.K(X)
    Kg = kern_g.K(X)
    Lf = GPy.util.linalg.jitchol(Kf)
    Lg = GPy.util.linalg.jitchol(Kg)
    #f = Lf.dot(np.random.randn(X.shape[0])[:, None])
    #g = Lg.dot(np.random.randn(X.shape[0])[:, None]) - 0.8 # With mean function
    #g = np.ones_like(f)*0.8

    #Use a fixed function
    f = 2*(np.exp(-30*(X[:,0] - 0.25)**2) + np.sin(np.pi*X[:,1]**2)) - 2
    g = np.sin(2*np.pi*X[:,0]) + np.cos(2*np.pi*X[:,1])

    f = f[:, None]
    g = g[:, None]

    #Link function to apply to f and g to ensure positiveness
    ef = np.exp(f)
    eg = np.exp(g)
    Y = np.asarray([sp.stats.fisk.rvs(c=egi, scale=efi)
                    for efi, egi in np.hstack((ef,eg))])[:, None]

    #Random censoring
    num_censoring = int((cens_perc/100.0)*Y.shape[0])
    random_censoring_inds = np.random.permutation(range(Y.shape[0]))[:num_censoring]
    censored_times = np.asarray([np.random.uniform(0, Yi) for Yi in Y[random_censoring_inds]])[:, None]
    Y[random_censoring_inds] = censored_times

    #Right censoring
    censor_val = 300.1
    right_censor_inds = Y > censor_val
    Y[right_censor_inds] = censor_val

    censoring = np.zeros_like(Y)
    censoring[random_censoring_inds] = 1
    censoring[right_censor_inds] = 1
    censoring = censoring.astype(bool)*1.0

    X = (X-X.mean(0))/X.std(0)
    Y_scale = float(sp.stats.gmean(Y))
    Y = (Y + 0.01) /Y_scale

    cv = cross_validation.KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=seed)
    #Stupidly you can't just index it...
    for i, (train, test) in enumerate(cv):
        if i==fold:
            training_inds = train
            test_inds = test

    # Remove censored from predictions
    uncensored_test_inds = censoring[test_inds, 0] < 0.5 # uncensored
    test_inds = test_inds[uncensored_test_inds]

    Xtrain = X[training_inds, :]
    Ytrain = Y[training_inds, :]
    Xtest = X[test_inds, :]
    Ytest = Y[test_inds, :]
    Y_metadata_train = {'censored': censoring[training_inds, :]}
    Y_metadata_test = {'censored': censoring[test_inds, :]}

    print "training shape: ", Xtrain.shape
    print "test shape: ", Xtest.shape
    print "All: ", X.shape
    print Xtrain.shape[0] + Xtest.shape[0] - X.shape[0]
    return Dataset(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, Ytest=Ytest,
                   Y_metadata_train=Y_metadata_train, Y_metadata_test=Y_metadata_test)


"""
Object to hold results of experiments
"""
class Experiment(object):
    def __init__(self, seed, fold, num_inducing, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, d_name, student_t, survival, optimize_df):
        #Make the three models, with different numbers of inducing points, and
        #with fixed and non_fixed #degrees of freedom
        self.seed, self.fold, self.num_inducing, self.fixZ, self.f_bias, self.g_bias, self.f_rbf_len, self.g_rbf_len, self.g_mean, self.f_rbf_var, self.d_name, self.optimize_df = seed, fold, num_inducing, fixZ, f_bias_var, g_bias_var, f_rbf_len, g_rbf_len, g_mean, f_rbf_var, d_name, optimize_df
        self.gauss_noise = gauss_noise

        np.random.seed(self.seed)

        if d_name == 'elevators1000':
            if self.num_inducing == 'all':
                self.num_inducing = 100 #Also if time do the same experiments with full GP
            self.dataset = load_elevators(fold=self.fold, seed=seed, num_training=1000)
            self.g_mean = np.log(0.1)
            random_func = random_multi_lengthscales_elevator
        elif d_name == 'elevators10000':
            if self.num_inducing == 'all':
                self.num_inducing = 100
            self.g_mean = np.log(0.1)
            self.dataset = load_elevators(fold=self.fold, seed=seed, num_training=10000)
            random_func = random_multi_lengthscales_elevator
        elif d_name == 'motorCorrupt':
            self.num_inducing = 'all'
            #A promising starting value
            self.g_mean = np.log(0.25)
            self.dataset = load_motorCorrupt(fold=self.fold, seed=seed)
            random_func = random_multi_lengthscales_motor
        elif d_name == 'boston':
            #self.num_inducing = 100
            self.g_mean = np.log(0.1)
            self.dataset = load_boston(fold=self.fold, seed=seed)
            random_func = random_multi_lengthscales_boston
        elif d_name == 'leukemia':
            self.num_inducing = 100
            self.g_mean = np.log(0.5)
            self.dataset = load_leukemia(fold=self.fold, seed=seed)
            random_func = random_multi_lengthscales_leukemia
        elif d_name == 'simulated_survival':
            self.num_inducing = 100
            self.g_mean = np.log(1.2)
            self.dataset = load_simulated_survival(fold=self.fold, seed=seed)
            random_func = random_multi_lengthscales_simulated_survival

        if self.num_inducing == 'all':
            self.Z = self.dataset.Xtrain.copy()
            self.fixZ = True
        else:
            self.Z = kmeans(self.dataset, self.num_inducing, seed=seed)

        #Randomize restarts starting with the suggested parameters
        m_gauss_pre_opt = preopt_gauss(self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, random_func, self.seed, self.fold)
        m_multi_gauss_pre_opt = preopt_multi_gauss(self.dataset, self.Z, self.fixZ, self.f_bias, self.g_bias, self.f_rbf_len, self.g_rbf_len, self.g_mean, self.f_rbf_var, random_func, self.seed, self.fold)
        if student_t:
            m_stut_pre_opt = preopt_stut(self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, random_func, self.seed, self.fold)
            m_laplace_stut_pre_opt = preopt_laplace_stut(self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, random_func, self.seed, self.fold)
            m_multi_stut_pre_opt = preopt_stut_multi(self.dataset, self.Z, self.fixZ, self.f_bias, self.g_bias, self.f_rbf_len, self.g_rbf_len, self.g_mean, self.f_rbf_var, random_func, self.seed, self.fold)
        if survival:
            m_survival_pre_opt = preopt_survival(self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, random_func, self.seed, self.fold)
            m_laplace_survival_pre_opt = preopt_laplace_survival(self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, random_func, self.seed, self.fold)
            m_multi_survival_pre_opt = preopt_survival_multi(self.dataset, self.Z, self.fixZ, self.f_bias, self.g_bias, self.f_rbf_len, self.g_rbf_len, self.g_mean, self.f_rbf_var, random_func, self.seed, self.fold)

        #Do full optimization with best models
        self.m_gauss_opt_params = optimize_gauss(m_gauss_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, self.seed, self.fold)
        self.m_multi_gauss_opt_params = optimize_multi_gauss(m_multi_gauss_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.g_bias, self.f_rbf_len, self.g_rbf_len, self.g_mean, self.f_rbf_var, self.seed, self.fold)
        if student_t:
            self.m_stut_opt_params = optimize_stut(m_stut_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, self.optimize_df, self.seed, self.fold)
            self.m_laplace_stut_opt_params = optimize_laplace_stut(m_laplace_stut_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, self.optimize_df, self.seed, self.fold)
            self.m_multi_stut_opt_params = optimize_multi_stut(m_multi_stut_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.g_bias, self.f_rbf_len, self.g_rbf_len, self.g_mean, self.f_rbf_var, self.optimize_df, self.seed, self.fold)
        else:
            self.m_stut_opt_params = None
            self.m_laplace_stut_opt_params = None
            self.m_multi_stut_opt_params = None
        if survival:
            self.m_survival_opt_params = optimize_survival(m_survival_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, self.seed, self.fold)
            self.m_laplace_survival_opt_params = optimize_laplace_survival(m_laplace_survival_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.f_rbf_len, self.f_rbf_var, self.seed, self.fold)
            self.m_multi_survival_opt_params = optimize_multi_survival(m_multi_survival_pre_opt, self.dataset, self.Z, self.fixZ, self.f_bias, self.g_bias, self.f_rbf_len, self.g_rbf_len, self.g_mean, self.f_rbf_var, self.seed, self.fold)
        else:
            self.m_survival_opt_params = None
            self.m_laplace_survival_opt_params = None
            self.m_multi_survival_opt_params = None

        results = evaluate(self.dataset, self.Z, self.fixZ,
                                          self.f_bias, self.g_bias,
                                          self.f_rbf_len, self.g_rbf_len,
                                          self.g_mean, self.f_rbf_var, self.seed, self.fold,
                                          self.m_gauss_opt_params,
                                          self.m_multi_gauss_opt_params,
                                          self.m_stut_opt_params, self.m_laplace_stut_opt_params, self.m_multi_stut_opt_params,
                                          self.m_survival_opt_params, self.m_laplace_survival_opt_params, self.m_multi_survival_opt_params,
                                          student_t=student_t, survival=survival)
        log_preds, RMSEs, MAEs, log_likelihoods, predictions, NMSEs = results[0], results[1], results[2], results[3], results[4], results[5]

        self.m_gauss_log_pred = log_preds[:, 0]
        self.m_gauss_NMSE = NMSEs[0]
        self.m_gauss_RMSE = RMSEs[0]
        self.m_gauss_MAE = MAEs[0]
        self.m_gauss_log_likelihood = log_likelihoods[0]
        self.m_gauss_predictions = predictions[0]

        self.m_multi_gauss_log_pred = log_preds[:, 1]
        self.m_multi_gauss_NMSE = NMSEs[1]
        self.m_multi_gauss_RMSE = RMSEs[1]
        self.m_multi_gauss_MAE = MAEs[1]
        self.m_multi_gauss_log_likelihood = log_likelihoods[1]
        self.m_multi_gauss_predictions = predictions[1]

        if student_t:
            self.m_stut_log_pred = log_preds[:, 2]
            self.m_stut_NMSE = NMSEs[2]
            self.m_stut_RMSE = RMSEs[2]
            self.m_stut_MAE = MAEs[2]
            self.m_stut_log_likelihood = log_likelihoods[2]
            self.m_stut_predictions = predictions[2]

            self.m_laplace_stut_log_pred = log_preds[:, 3]
            self.m_laplace_stut_NMSE = NMSEs[3]
            self.m_laplace_stut_RMSE = RMSEs[3]
            self.m_laplace_stut_MAE = MAEs[3]
            self.m_laplace_stut_log_likelihood = log_likelihoods[3]
            self.m_laplace_stut_predictions = predictions[3]

            self.m_multi_stut_log_pred = log_preds[:, 4]
            self.m_multi_stut_NMSE = NMSEs[4]
            self.m_multi_stut_RMSE = RMSEs[4]
            self.m_multi_stut_MAE = MAEs[4]
            self.m_multi_stut_log_likelihood = log_likelihoods[4]
            self.m_multi_stut_predictions = predictions[4]
        else:
            (self.m_stut_log_pred, self.m_laplace_stut_log_pred, self.m_multi_stut_log_pred,
            self.m_stut_NMSE, self.m_laplace_stut_NMSE, self.m_multi_stut_NMSE,
            self.m_stut_RMSE, self.m_laplace_stut_RMSE, self.m_multi_stut_RMSE,
            self.m_stut_MAE, self.m_laplace_stut_MAE, self.m_multi_stut_MAE,
            self.m_stut_log_likelihood, self.m_laplace_stut_log_likelihood, self.m_multi_stut_log_likelihood,
            self.m_stut_predictions, self.m_laplace_stut_predictions, self.m_multi_stut_predictions) = [None]*18

        if survival:
            self.m_survival_log_pred = log_preds[:, 5]
            self.m_survival_NMSE = NMSEs[5]
            self.m_survival_RMSE = RMSEs[5]
            self.m_survival_MAE = MAEs[5]
            self.m_survival_log_likelihood = log_likelihoods[5]
            self.m_survival_predictions = predictions[5]

            self.m_laplace_survival_log_pred = log_preds[:, 6]
            self.m_laplace_survival_NMSE = NMSEs[6]
            self.m_laplace_survival_RMSE = RMSEs[6]
            self.m_laplace_survival_MAE = MAEs[6]
            self.m_laplace_survival_log_likelihood = log_likelihoods[6]
            self.m_laplace_survival_predictions = predictions[6]

            self.m_multi_survival_log_pred = log_preds[:, 7]
            self.m_multi_survival_NMSE = NMSEs[7]
            self.m_multi_survival_RMSE = RMSEs[7]
            self.m_multi_survival_MAE = MAEs[7]
            self.m_multi_survival_log_likelihood = log_likelihoods[7]
            self.m_multi_survival_predictions = predictions[7]
        else:
            (self.m_survival_log_pred, self.m_laplace_survival_log_pred, self.m_multi_survival_log_pred,
            self.m_survival_NMSE, self.m_laplace_survival_NMSE, self.m_multi_survival_NMSE,
            self.m_survival_RMSE, self.m_laplace_survival_RMSE, self.m_multi_survival_RMSE,
            self.m_survival_MAE, self.m_laplace_survival_MAE, self.m_multi_survival_MAE,
            self.m_survival_log_likelihood, self.m_laplace_survival_log_likelihood, self.m_multi_survival_log_likelihood,
            self.m_survival_predictions, self.m_laplace_survival_predictions, self.m_multi_survival_predictions) = [None]*18

    def __str__(self):
        return "Fold: {}, Seed: {}, M: {}, fixZ: {}, f_bias: {}, g_bias: {}".format(self.fold, self.seed, self.num_inducing, self.fixZ, self.f_bias, self.g_bias)

    def __repr__(self):
        return "Fold: {}, Seed: {}, M: {}, fixZ: {}, f_bias: {}, g_bias: {}".format(self.fold, self.seed, self.num_inducing, self.fixZ, self.f_bias, self.g_bias)

gauss_experiments = None
stut_experiments = None
survival_experiments = None
if len(survival_dataset_names) > 0:
    survival_experiments = [Experiment(s, f, M, fZ, f_b, g_b, f_l, g_l, g_m, f_v, d_name, student_t=False, survival=True, optimize_df=False) for f in folds for M in Ms for fZ in fix_Zs for f_b in f_bias_vars for g_b in g_bias_vars for f_l in f_rbf_lens for g_l in g_rbf_lens for g_m in g_means for s in seeds for f_v in f_rbf_vars for d_name in survival_dataset_names]
if len(gauss_dataset_names) > 0:
    gauss_experiments    = [Experiment(s, f, M, fZ, f_b, g_b, f_l, g_l, g_m, f_v, d_name, student_t=False, survival=False, optimize_df=False) for f in folds for M in Ms for fZ in fix_Zs for f_b in f_bias_vars for g_b in g_bias_vars for f_l in f_rbf_lens for g_l in g_rbf_lens for g_m in g_means for s in seeds for f_v in f_rbf_vars for d_name in gauss_dataset_names]
if len(stut_dataset_names) > 0:
    stut_experiments     = [Experiment(s, f, M, fZ, f_b, g_b, f_l, g_l, g_m, f_v, d_name, student_t=True, survival=False, optimize_df=o_df) for f in folds for M in Ms for fZ in fix_Zs for f_b in f_bias_vars for g_b in g_bias_vars for f_l in f_rbf_lens for g_l in g_rbf_lens for g_m in g_means for s in seeds for f_v in f_rbf_vars for d_name in stut_dataset_names for o_df in optimize_dfs]
