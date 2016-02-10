# Copyright (c) 2015 Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import scipy as sp
from GPy.likelihoods import link_functions
from GPy.util.misc import safe_exp
from multi_likelihood import MultiLikelihood
from functools import partial

class MultiPoisson(MultiLikelihood):
    """
    Poisson likelihood with sum of latent functions for rate
    """
    def __init__(self,gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Log()

        super(MultiPoisson, self).__init__(gp_link, name='Multi_poisson')
        self.log_concave = False
        self.run_already = False
        self.T = 15  # number of points to use in quadrature

    def request_num_latent_functions(self, Y):
        """
        The likelihood should infer how many latent functions are needed for the likelihood
        In this case we have one latent function for both shape parameters, for each output dimension
        """
        return Y.shape[1]*2

    def pdf(self, f, g, y, Y_metadata=None):
        return safe_exp(self.logpdf(f,y,Y_metadata=Y_metadata))

    def logpdf(self, f, y, Y_metadata=None):
        """
        Log Likelihood Function given f

        .. math::
            \\ln p(y_{i}|f_{i}, g_{i}) = -(\\lambda(f_{i}) + \\lambda(g_{i}))\\ + y_{i}\\log (\\lambda(f_{i}) + \\lambda(g_{i})) - \\log y_{i}!

        :param link_f: latent variables (link(f))
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        D = y.shape[1]
        fv, gv = f[:, :D], f[:, D:]
        link_f = safe_exp(fv)
        link_g = safe_exp(gv)
        return -(link_f + link_g) + y*np.logaddexp(fv, gv) - sp.special.gammaln(y+1)

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        pass

    def predictive_mean(self, mu, sigma, Y_metadata=None):
        raise NotImplementedError

    def predictive_variance(self, mu,variance, predictive_mean=None, Y_metadata=None):
        raise NotImplementedError

    def conditional_mean(self, gp):
        raise NotImplementedError

    def conditional_variance(self, gp):
        raise NotImplementedError

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        if not self.run_already:
            from theano import tensor as t
            import theano
            y = t.matrix(name='y')
            f = t.matrix(name='f')
            g = t.matrix(name='g')

            def theano_logaddexp(x,y):
                #Implementation of logaddexp from numpy, but in theano
                tmp = x - y
                return t.where(tmp > 0, x + t.log1p(t.exp(-tmp)), y + t.log1p(t.exp(tmp)))

            # Full log likelihood before expectations
            logpy_t = -(t.exp(f) + t.exp(g)) + y*theano_logaddexp(f, g) - t.gammaln(y+1)
            logpy_sum_t = t.sum(logpy_t)

            dF_df_t = theano.grad(logpy_sum_t, f)
            d2F_df2_t = 0.5*theano.grad(t.sum(dF_df_t), f)  # This right?
            dF_dg_t = theano.grad(logpy_sum_t, g)
            d2F_dg2_t = 0.5*theano.grad(t.sum(dF_dg_t), g)  # This right?

            self.logpy_func = theano.function([f,g,y],logpy_t)
            self.dF_df_func = theano.function([f,g,y],dF_df_t)  # , mode='DebugMode')
            self.d2F_df2_func = theano.function([f,g,y],d2F_df2_t)
            self.dF_dg_func = theano.function([f,g,y],dF_dg_t)
            self.d2F_dg2_func = theano.function([f,g,y],d2F_dg2_t)
            self.run_already = True

        funcs = [self.logpy_func, self.dF_df_func, self.d2F_df2_func, self.dF_dg_func, self.d2F_dg2_func]

        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        F = 0  # Could do analytical components here

        T = self.T
        # Need to get these now to duplicate the censored inputs for quadrature
        gh_x, gh_w = self._gh_points(T)

        (F_quad, dF_dmf, dF_dvf, dF_dmg, dF_dvg) = self.quad2d(funcs=funcs, Y=Y, mf=mf, vf=vf, mg=mg, vg=vg,
                                                               gh_points=gh_points, exp_f=False, exp_g=False)

        F += F_quad
        # gprec = safe_exp(mg - 0.5*vg)
        dF_dmf += 0
        dF_dmg += 0
        dF_dvf += 0
        dF_dvg += 0

        dF_dm = np.hstack((dF_dmf, dF_dmg))
        dF_dv = np.hstack((dF_dvf, dF_dvg))

        if np.any(np.isnan(F_quad)):
            raise ValueError("Nan <log p(y|f,g)>_qf_qg")
        if np.any(np.isnan(dF_dmf)):
            raise ValueError("Nan gradients <log p(y|f,g)>_qf_qg wrt to qf mean")
        if np.any(np.isnan(dF_dmg)):
            raise ValueError("Nan gradients <log p(y|f,g)>_qf_qg wrt to qg mean")

        test_integration = False
        if test_integration:
            # Some little code to check the result numerically using quadrature
            from scipy import integrate
            i = 6  # datapoint index

            def quad_func(fi, gi, yi, mgi, vgi, mfi, vfi):
                #link_fi = np.exp(fi)
                #link_gi = np.exp(gi)
                #logpy_fg = -(link_fi + link_gi) + yi*np.logaddexp(fi, gi) - sp.special.gammaln(yi+1)
                logpy_fg = self.logpdf(np.atleast_2d(np.hstack([fi,gi])), np.atleast_2d(yi))
                return (logpy_fg  # log p(y|f,g)
                        * np.exp(-0.5*np.log(2*np.pi*vgi) - 0.5*((gi - mgi)**2)/vgi)  # q(g)
                        * np.exp(-0.5*np.log(2*np.pi*vfi) - 0.5*((fi - mfi)**2)/vfi)  # q(f)
                        )
            quad_func_l = partial(quad_func, yi=Y[i], mgi=mg[i], vgi=vg[i], mfi=mf[i], vfi=vf[i])

            def integrl(gi):
                return integrate.quad(quad_func_l, -70, 70, args=(gi))[0]

            print "These should match"

            print "Numeric scipy F quad"
            print integrate.quad(lambda fi: integrl(fi), -70, 70)

            print "2d quad F quad"
            print F[i]

        return F, dF_dm, dF_dv, None
