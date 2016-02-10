# Copyright (c) 2015 James Hensman, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from GPy.util.univariate_Gaussian import std_norm_pdf, std_norm_cdf
from GPy.likelihoods import link_functions
#from GPy.likelihoods.likelihood import Likelihood
from multi_likelihood import MultiLikelihood

class HetGP(MultiLikelihood):
    """
    Hetroschedastic GP where we have two latent functions, one over the mean, one over the variance
    """
    def __init__(self, gp_link=None):
        if gp_link is not None:
            raise NotImplementedError, "this likelihood assumes a complicated pair of link functions..."

        super(HetGP, self).__init__(link_functions.Identity(), 'HetGP')

    def request_num_latent_functions(self, Y):
        """
        The likelihood should infer how many latent functions are needed for the likelihood

        In this case we have one latent function for mean and one for scale, for each output dimension
        """
        return Y.shape[1]*2

    def conditional_mean(self, gp, Y_metadata):
        pass

    def logpdf(self, f, y, Y_metadata=None):
        D = y.shape[1]
        fv, gv = f[:, :D], f[:, D:]
        e_g = np.exp(gv)
        y_f2 = (y-fv)**2
        return -0.5*np.log(2*np.pi) - 0.5*gv - 0.5*(y_f2/e_g)

    def pdf(self, f, g, y, Y_metadata):
        """
        .. math::
            \\p(y_{i}|\\f_{i},\\g_{i}) = \\frac{1}{2pi\\exp(g_{i}}\\exp\\big(\\frac{(y_{i} - f_{i})^{2}}{2\\exp(g_{i}}\\big)
        """
        e_g = np.exp(g)
        y_f2 = (y-f)**2
        pdf = np.exp(-0.5*np.log(2*np.pi*e_g) -0.5*(y_f2/e_g))
        return pdf

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        D = gp.shape[1]/2
        f, g = gp[:,:D], gp[:,D:]
        std = np.sqrt(np.exp(g))
        return np.random.randn(gp.shape[0], D)*std + f

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        """
        E_{q(f)q(g)}[log p(y|f,g)]
        Result in analytic in the Gaussian case
        """
        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        precision = np.exp(-mg + 0.5*vg)
        squares = (np.square(Y) + np.square(mf) + vf - 2*mf*Y)
        F = -0.5*np.log(2*np.pi) - 0.5*mg - 0.5*precision*squares
        dF_dmf = precision*(Y-mf)
        dF_dmg = 0.5*(precision*squares - 1.)
        dF_dvf = -0.5*precision
        dF_dvg = -0.25*precision*squares

        dF_dm = np.hstack((dF_dmf, dF_dmg))
        dF_dv = np.hstack((dF_dvf, dF_dvg))
        return F, dF_dm, dF_dv, None
