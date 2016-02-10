from GPy.likelihoods.likelihood import Likelihood
import numpy as np
from scipy.misc import logsumexp
from GPy.util.misc import safe_exp

class MultiLikelihood(Likelihood):
    def __init__(self, gp_link, name):
        super(MultiLikelihood, self).__init__(gp_link=gp_link, name=name)

    def log_predictive_density_sampling(self, y_test, mu_star, var_star, Y_metadata=None, num_samples=1000):
        """
        Calculation of the log predictive density via sampling

        .. math:
            log p(y_{*}|D) = log \frac{1}{num_samples} \prod^{S}_{s=1} p(y^{*}|f_{s}^{*},g_{s}^{*})
            f_{s}^{*} ~ p(f^{*}|m^{*}_{f},v^{*}_{f})
            g_{s}^{*} ~ p(g^{*}|m^{*}_{g}, v^{*}_{g})

        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param mu_stars: predictive means of gaussians, p(f^{*}|m^{*}_{f},v^{*}_{f}) and p(g^{*}|m^{*}_{g}, v^{*}_{g})
        :type mu_stars: (Nx1) array
        :param var_stars: predictive variances of gaussians p(f^{*}|m^{*}_{f},v^{*}_{f}) and p(g^{*}|m^{*}_{g}, v^{*}_{g})
        :type var_stars: (Nx1) array
        :param num_samples: number of samples to use in the monte carlo integration
        :type num_samples: int
        """
        num_latent_funcs = self.request_num_latent_functions(y_test)
        assert mu_star.shape[1]/y_test.shape[1] == num_latent_funcs

        #Take samples of p(f*|y)
        num_pred, num_outputs = y_test.shape
        fi_samples = np.empty((num_pred, num_outputs*num_latent_funcs, num_samples))
        for lf in range(num_latent_funcs):
            mu_star_f = mu_star[:, lf*num_outputs:(lf+1)*num_outputs][:, None]
            var_star_f = var_star[:, lf*num_outputs:(lf+1)*num_outputs][:, None]
            fi_samples[:, lf*num_outputs:(lf+1)*num_outputs, :] = np.random.normal(mu_star_f, np.sqrt(var_star_f), size=(num_pred, num_outputs, num_samples))

        #Needs an extra dimension to broadcast properly
        y_broad = y_test[:, None]
        log_p_ystar = -np.log(num_samples) + logsumexp(self.logpdf(fi_samples, y_broad, Y_metadata=Y_metadata), axis=-1)
        log_p_ystar = np.array(log_p_ystar).reshape(*y_test.shape)
        return log_p_ystar

    def log_predictive_density(self, y_test, mu_stars, var_stars, Y_metadata=None):
        """
        Calculation of the log predictive density

        .. math:
            p(y^{*}|D) = p(y^{*}|f^{*},g^{*}).p(f^{*}|m^{*}_{f},v^{*}_{f}).p(g^{*}|m^{*}_{g}, v^{*}_{g})


        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param mu_stars: predictive means of gaussians, p(f^{*}|m^{*}_{f},v^{*}_{f}) and p(g^{*}|m^{*}_{g}, v^{*}_{g})
        :type mu_stars: (Nx1) array
        :param var_stars: predictive variances of gaussians p(f^{*}|m^{*}_{f},v^{*}_{f}) and p(g^{*}|m^{*}_{g}, v^{*}_{g})
        :type var_stars: (Nx1) array
        """
        assert y_test.shape==mu_stars[0].shape
        assert y_test.shape==var_stars[0].shape
        assert y_test.shape==mu_stars[1].shape
        assert y_test.shape==var_stars[1].shape
        #1 output only
        assert y_test.shape[1] == 1
        #Only working for two latent functions at the minute, try sampling otherwise!
        assert self.request_num_latent_functions(y_test) == 2

        mf = mu_stars[0]
        mg = mu_stars[1]
        vf = var_stars[0]
        vg = var_stars[1]

        pdf_partial = self.pdf_partial(Y_metadata=Y_metadata)

        p_ystar = self.quad2d([pdf_partial], y_test, mf, vf, mg, vg, gh_points=None, exp_g=False)
        p_ystar = np.array(p_ystar).reshape(*y_test.shape)
        return np.log(p_ystar)

    def quad2d(self, funcs, Y, mf, vf, mg, vg, gh_points, exp_f=False, exp_g=False, **kwargs):
        """
        Given a list of functions to do quadrature over, expecting just latent functions
        f, and g to integrate over, do the quadrature for each and return a list of results

        Requires the list of functions, Y and
        the mean and variances of the two distributions to integrate over

        exp_g takes the exponential of g before passing it to ALL funcs in the list, be careful that this is
        what you want
        """
        #Do some testing to see if the quadrature works well for one datapoint
        if gh_points is None:
            gh_x, gh_w = self._gh_points(T=10)
        else:
            gh_x, gh_w = gh_points

        #Numpy implementation of quadrature
        #We want to make it so we have the same shapes before we flatten them, so we make space for everything
        Xf = np.zeros((mf.shape[0], gh_x.shape[0], gh_x.shape[0], mf.shape[1]))
        Xg = np.zeros((mg.shape[0], gh_x.shape[0], gh_x.shape[0], mg.shape[1]))
        #Need to stretch the points
        Xf[:] = gh_x[None, :, None, None]*np.sqrt(2*vf[:, None, None, :]) + mf[:, None, None, :]
        Xg[:] = gh_x[None, None, :, None]*np.sqrt(2*vg[:, None, None, :]) + mg[:, None, None, :]
        #def f_func(f, g, y):
            #return -0.5*np.log(2*np.pi*np.exp(g)) -0.5*((y-f)**2)/np.exp(g)

        #Reshape into a big array ready for the function to be applied such that it broadcasts properly
        Xf_rs = Xf.reshape(Xf.shape[0]*Xf.shape[1]*Xf.shape[2], -1, order='C')
        Xg_rs = Xg.reshape(Xg.shape[0]*Xg.shape[1]*Xg.shape[2], -1, order='C')
        y_full = np.repeat(Y, gh_x.shape[0]**2, axis=0)
        results = []

        if exp_g:
            Xg_rs = safe_exp(Xg_rs)

        if exp_f:
            Xf_rs = safe_exp(Xf_rs)

        for func in funcs:
            func_res= func(Xf_rs, Xg_rs, y_full, **kwargs)
            func_res = func_res.reshape(mf.shape[0], gh_x.shape[0], gh_x.shape[0])
            #division by pi comes from fact that for each quadrature we need to scale by 1/sqrt(pi)
            #Assume 1D out at the moment
            quad_result = (np.dot(func_res, gh_w).dot(gh_w)/np.pi)[:, None]
            results.append(quad_result)

            #Could maybe do -?
            #quad_result = func*gh_w[None, :, None]*gh_w[None, None, :]
            #quad_result= quad_result.sum(-1).sum(-1)/np.pi

        return results

    def pdf_partial(self, Y_metadata):
        """
        Should be overriden for models with parameters that are fixed throughout a quadrature
        """
        from functools import partial
        return partial(self.pdf, Y_metadata=Y_metadata)
