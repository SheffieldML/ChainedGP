# Copyright (c) 2015 Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
#from scipy import stats, integrate
from GPy.likelihoods import link_functions
from GPy.util.misc import safe_exp, safe_square
#from GPy.likelihoods.likelihood import Likelihood
from multi_likelihood import MultiLikelihood
#import quad_cython
from functools import partial

class HetLogLogistic(MultiLikelihood):
    """
    LogLogistic likelihood with latent functions over both median and shape
    """
    def __init__(self,gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Log()

        super(HetLogLogistic, self).__init__(gp_link, name='Het_loglogistic')
        self.log_concave = False
        self.run_already = False
        self.T = 10  # number of points to use in quadrature

    def request_num_latent_functions(self, Y):
        """
        The likelihood should infer how many latent functions are needed for the likelihood
        In this case we have one latent function for both shape parameters, for each output dimension
        """
        return Y.shape[1]*2

    def pdf(self, f, y, Y_metadata=None):
        return np.exp(self.logpdf(f, y, Y_metadata=Y_metadata))

    def logpdf(self, f, y, Y_metadata=None):
        c = Y_metadata['censored']
        c = c.reshape(*y.shape)
        D = y.shape[1]
        fv, gv = f[:, :D], f[:, D:]
        ef = safe_exp(fv)
        eg = safe_exp(gv)
        #y_link_f_r = np.clip((y/ef)**eg, 1e-150, 1e200)
        # y_link_f_r = (y/ef)**eg
        # log1p_y_link_f_r = np.log1p(y_link_f_r)

        def log1pexp(x):
            # more Numerically stable "softplus"
            b1 = np.atleast_1d(x <= -37)
            b2 = np.atleast_1d((x > -37) & (x <= 18))
            b3 = np.atleast_1d((x > 18) & (x <= 33.3))
            b4 = np.atleast_1d(x > 33.3)
            xc = np.atleast_1d(x)
            xnew = np.zeros_like(x)*np.nan
            xnew = np.atleast_1d(xnew)
            xnew[b1] = np.exp(xc[b1])
            xnew[b2] = np.log1p(np.exp(xc[b2]))
            xnew[b3] = xc[b3] + np.exp(-xc[b3])
            xnew[b4] = xc[b4]
            if np.isscalar(x):
                return float(xnew)
            return xnew

        # log1p_y_link_f_r = np.log1p(np.exp(eg*(np.log(y) - fv)))
        log1p_y_link_f_r = log1pexp(eg*(np.log(y) - fv))
        uncensored = (1-c)*(np.log(eg) + (eg-1)*np.log(y) - eg*np.log(ef) - 2*log1p_y_link_f_r)
        censored = (c)*(-log1p_y_link_f_r)
        return uncensored + censored

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        pass

    def predictive_mean(self, mu, sigma, Y_metadata=None):
        # The comment here confuses mean and median.
        raise NotImplementedError
        #return self.gp_link.transf(mu) # only true if link is monotonic, which it is.

    def predictive_variance(self, mu,variance, predictive_mean=None, Y_metadata=None):
        raise NotImplementedError

    def conditional_mean(self, gp):
        #return self.gp_link.transf(gp)
        raise NotImplementedError

    def conditional_variance(self, gp):
        #Expects just g!
        raise NotImplementedError

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        raise NotImplementedError
        orig_shape = gp.shape
        gp = gp.flatten()
        #Can't get it to sample all at the same time
        #loglogistic_samples = np.array([stats.fisk.rvs(self.gp_link.transf(gpf), self.gp_link.transf(gpg), size=1) for gpf,gpg in gp])
        return loglogistic_samples.reshape(orig_shape)

    def variational_expectations_pure(self, Y, m, v, gh_points=None, Y_metadata=None):

        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        from GPy.util.misc import safe_exp, safe_square

        c = Y_metadata['censored']
        #ef = safe_exp(f)
        #eg = safe_exp(g)
        emg_vg2 = safe_exp(mg - 0.5*vg)
        uncensored = (1-c)*(mg + (emg_vg2 - 1)*np.log(Y) - mf*emg_vg2)
        censored = (c)*(0)
        F = 0 # uncensored + censored

        T = 20
        #Need to get these now to duplicate the censored inputs for quadrature
        gh_x, gh_w = self._gh_points(T)
        Y_metadata_new = Y_metadata.copy()
        Y_metadata_new['censored'] = np.repeat(Y_metadata_new['censored'], gh_x.shape[0]**2, axis=0)

        ##Some little code to check the result numerically using quadrature
        #from scipy import integrate
        #i = 6  # datapoint index
        #def quad_func(fi, gi, yi, mgi, vgi, mfi, vfi,ci):
            ##x = safe_exp(-fi*safe_exp(gi))*yi**safe_exp(gi)
            #x = safe_exp(-fi*safe_exp(gi) + safe_exp(gi)*np.log(yi))
            #log1px = np.log1p(x)
            ##return ((*-gammaln(np.exp(fi)) - gammaln(np.exp(gi)) + gammaln(np.exp(fi) + np.exp(gi)))      #p(y|f,g)
            #return (((1-ci)*(-2*log1px) + ci*(-log1px))      #p(y|f,g)
                    #* np.exp(-0.5*np.log(2*np.pi*vgi) - 0.5*((gi - mgi)**2)/vgi) #q(g)
                    #* np.exp(-0.5*np.log(2*np.pi*vfi) - 0.5*((fi - mfi)**2)/vfi) #q(f)
                    #)
        #quad_func_l = partial(quad_func, yi=Y[i], mgi=mg[i], vgi=vg[i], mfi=mf[i], vfi=vf[i], ci=Y_metadata['censored'][i])
        #def integrl(gi):
            #return integrate.quad(quad_func_l, -30, 5, args=(gi))[0]
        #print "Numeric scipy F quad"
        #print integrate.quad(lambda fi: integrl(fi), -30, 5)

        def F_quad_func(f, e_g, y, Y_metadata):
            c = Y_metadata['censored']
            x = safe_exp(-f*e_g + e_g*np.log(y))
            log1px = np.log1p(x)
            uncensored = (1-c)*(-2*log1px)
            censored = c*(-log1px)
            return uncensored + censored

        def F_dquad_df_func(f, e_g, y, Y_metadata):
            c = Y_metadata['censored']
            x = e_g*(1./(safe_exp(e_g*(-f + np.log(y))) + 1) - 1)
            uncensored = (1-c)*(-2*x)
            censored = c*(-x)
            return uncensored + censored

        def F_d2quad_df2_func(f, e_g, y, Y_metadata):
            c = Y_metadata['censored']
            l_y = np.log(y)
            x = safe_exp(e_g*(l_y - f))
            e2g = np.exp(2*np.log(e_g))
            t = e2g/(x+1) - e2g/(x+1)**2
            uncensored = (1-c)*(-t)
            censored = c*(-0.5*t)
            return uncensored + censored

        def F_dquad_dg_func(f, e_g, y, Y_metadata):
            c = Y_metadata['censored']
            l_y = np.log(y)
            x = safe_exp(e_g*(l_y - f))
            denom = x + 1
            a = e_g*(l_y - f)
            t = a - a/denom
            uncensored = (1-c)*(-2*t)
            censored = c*(-t)
            return uncensored + censored

        def F_d2quad_dg2_func(f, e_g, y, Y_metadata):
            return np.zeros_like(y)  # ?

        (F_quad, dF_dmf, dF_dvf,
        dF_dmg, dF_dvg) = self.quad2d([F_quad_func, F_dquad_df_func, F_d2quad_df2_func, F_dquad_dg_func, F_d2quad_dg2_func],
                                              Y, mf, vf, mg, vg, gh_points, exp_f=False, exp_g=True, Y_metadata=Y_metadata_new)
        #F_quad = self.quad2d([F_quad_func], Y, mf, vf, mg, vg, gh_points, exp_f=False, exp_g=True)[0]
        #dF_dmf = self.quad2d([F_dquad_df_func], Y, mf, vf, mg, vg, gh_points, exp_f=False, exp_g=True)[0]
        #dF_dvf = self.quad2d([F_d2quad_df2_func], Y, mf, vf, mg, vg, gh_points, exp_f=False, exp_g=True)[0]
        #dF_dmg = self.quad2d([F_dquad_dg_func], Y, mf, vf, mg, vg, gh_points, exp_f=False, exp_g=True)[0]
        #dF_dvg = self.quad2d([F_d2quad_dg2_func], Y, mf, vf, mg, vg, gh_points, exp_f=False, exp_g=True)[0]

        #print "2d quad F quad"
        #print F_quad[i]
        F += F_quad
        gprec = safe_exp(mg - 0.5*vg)
        dF_dmf += 0 #(1-c)*(-gprec)
        dF_dmg += 0 #(1-c)*(1 + gprec*(np.log(Y) - mf))
        dF_dvf += 0
        dF_dvg += 0  # ?

        dF_dm = np.hstack((dF_dmf, dF_dmg))
        dF_dv = np.hstack((dF_dvf, dF_dvg))

        ##CYTHON
        ##Do some testing to see if the quadrature works well for one datapoint
        #Ngh = 20
        #if gh_points is None:
            #gh_x, gh_w = self._gh_points(T=Ngh)
        #else:
            #gh_x, gh_w = gh_points

        #F_quad_cython = np.zeros(Y.shape)
        #dF_dmg_cython = np.zeros(mg.shape)
        #dF_dmf_cython = np.zeros(mf.shape)
        #dF_dvf_cython = np.zeros(vf.shape)
        #dF_dvg_cython = np.zeros(vg.shape)
        #dF_ddf_cython = np.zeros(vg.shape)
        #quad_cython.quad2d_loglogistic(mf, vf, mg, vg, Y, gh_x, gh_w, F_quad_cython,
                                  #dF_dmf_cython, dF_dvf_cython, dF_dmg_cython,
                                  #dF_dvg_cython)

        #F_quad_cython /= np.pi
        #dF_dmg_cython /= np.pi
        #dF_dmf_cython /= np.pi
        #dF_dvf_cython /= np.pi
        #dF_dvg_cython /= np.pi
        #dF_ddf_cython /= np.pi

        return F, dF_dm, dF_dv, None

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        if not self.run_already:
            from theano import tensor as t
            import theano
            #Should really be a matrix for multiple outputs
            y = t.matrix(name='y')
            f = t.matrix(name='f')
            g = t.matrix(name='g')
            c = t.matrix(name='c')
            #ef = t.where(f > 18., f, t.log1p(t.exp(f)))
            #eg = t.where(g > 18., g, t.log1p(t.exp(g)))
            #ef = t.nnet.softplus(f)
            #eg = t.nnet.softplus(g)
            ef = t.exp(f)
            eg = t.exp(g)

            #In log(1+b) if b > 300, use log(b) as 1 isn't relevant anymore
            #inner_1 = (y/ef)**eg  # Naively
            #inner = t.exp(eg*(t.log(y) - t.log(ef)))  # do it in log space then exp, then do log1p
            #clip_log1p_inner_1 = t.where(inner_1 > 300, eg*(t.log(y) - t.log(ef)), t.log1p(inner_1))
            #clip_log1p_inner = t.log1p(inner)

            inner = eg*(t.log(y) - t.log(ef))  # We are going to do log(1+a) which is log(1+exp(log a)) which is softplus(log a) where log a is stable!
            clip_log1p_inner = t.nnet.softplus(inner)

            #Full log likelihood before expectations
            #logpy_t = (1-c)*(+t.log(eg) - eg*t.log(ef) + (eg - 1)*t.log(y) - 2*clip_log1p_inner_1) + c*(-clip_log1p_inner_1)
            #logpy_t_1 = t.where(c, -clip_log1p_inner_1, t.log(eg) - eg*t.log(ef) + (eg - 1)*t.log(y) - 2*clip_log1p_inner_1)
            logpy_t = t.where(c, -clip_log1p_inner, t.log(eg) - eg*t.log(ef) + (eg - 1)*t.log(y) - 2*clip_log1p_inner)
            logpy_sum_t = t.sum(logpy_t)

            dF_df_t = theano.grad(logpy_sum_t, f)
            d2F_df2_t = 0.5*theano.grad(t.sum(dF_df_t), f)  # This right?
            dF_dg_t = theano.grad(logpy_sum_t, g)
            d2F_dg2_t = 0.5*theano.grad(t.sum(dF_dg_t), g)  # This right?

            self.logpy_func = theano.function([f,g,y,c],logpy_t)
            self.dF_df_func = theano.function([f,g,y,c],dF_df_t)#, mode='DebugMode')
            self.d2F_df2_func = theano.function([f,g,y,c],d2F_df2_t)
            self.dF_dg_func = theano.function([f,g,y,c],dF_dg_t)
            self.d2F_dg2_func = theano.function([f,g,y,c],d2F_dg2_t)
            self.run_already = True

        #funcs = [self.logpy_func, self.dF_df_func, self.d2F_df2_func, self.dF_dg_func, self.d2F_dg2_func]
        funcs = [self.logpy_func, self.dF_df_func, self.d2F_df2_func, self.dF_dg_func, self.d2F_dg2_func]

        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        c = Y_metadata['censored']
        F = 0 # Could do analytical components here

        T = self.T
        #Need to get these now to duplicate the censored inputs for quadrature
        gh_x, gh_w = self._gh_points(T)
        Y_metadata_new= Y_metadata.copy()
        c = np.repeat(Y_metadata_new['censored'], gh_x.shape[0]**2, axis=0)

        ##Some little code to check the result numerically using quadrature
        #from scipy import integrate
        #i = 6  # datapoint index
        #def quad_func(fi, gi, yi, mgi, vgi, mfi, vfi,ci):
            ##x = safe_exp(-fi*safe_exp(gi))*yi**safe_exp(gi)
            #x = safe_exp(-fi*safe_exp(gi) + safe_exp(gi)*np.log(yi))
            #log1px = np.log1p(x)
            ##return ((*-gammaln(np.exp(fi)) - gammaln(np.exp(gi)) + gammaln(np.exp(fi) + np.exp(gi)))      #p(y|f,g)
            #return (((1-ci)*(-2*log1px) + ci*(-log1px))      #p(y|f,g)
                    #* np.exp(-0.5*np.log(2*np.pi*vgi) - 0.5*((gi - mgi)**2)/vgi) #q(g)
                    #* np.exp(-0.5*np.log(2*np.pi*vfi) - 0.5*((fi - mfi)**2)/vfi) #q(f)
                    #)
        #quad_func_l = partial(quad_func, yi=Y[i], mgi=mg[i], vgi=vg[i], mfi=mf[i], vfi=vf[i], ci=Y_metadata['censored'][i])
        #def integrl(gi):
            #return integrate.quad(quad_func_l, -30, 5, args=(gi))[0]
        #print "Numeric scipy F quad"
        #print integrate.quad(lambda fi: integrl(fi), -30, 5)

        #(F_quad, dF_dmf, dF_dvf, dF_dmg, dF_dvg) = self.quad2d(funcs=funcs, Y=Y, mf=mf, vf=vf, mg=mg, vg=vg,
                                                               #gh_points=gh_points, exp_f=False, exp_g=False, c=c)
        (F_quad, dF_dmf, dF_dvf, dF_dmg, dF_dvg) = self.quad2d(funcs=funcs, Y=Y, mf=mf, vf=vf, mg=mg, vg=vg,
                                                               gh_points=gh_points, exp_f=False, exp_g=False, c=c)

        #print "2d quad F quad"
        #print F_quad[i]
        F += F_quad
        #gprec = safe_exp(mg - 0.5*vg)
        dF_dmf += 0  #(1-c)*(-gprec)
        dF_dmg += 0  #(1-c)*(1 + gprec*(np.log(Y) - mf))
        dF_dvf += 0  # ?
        dF_dvg += 0  # ?

        dF_dm = np.hstack((dF_dmf, dF_dmg))
        dF_dv = np.hstack((dF_dvf, dF_dvg))

        if np.any(np.isnan(F_quad)):
            print("We have a nan in F_quad")
        if np.any(np.isnan(dF_dmf)):
            print("We have a nan in dF_dmf")
        if np.any(np.isnan(dF_dmg)):
            print("We have a nan in dF_dmg")

        return F, dF_dm, dF_dv, None

    #def pdf_partial(self, Y_metadata):
        #"""
        #Should be overriden for models with parameters that are fixed throughout a quadrature
        #"""
        #from functools import partial
        #df = float(self.deg_free[:])
        #return partial(self.pdf, df=df, Y_metadata=Y_metadata)

    def quad2D_weave(self, f, g, y, v, gh_w, f_string, e_g, D):
        #Broken!
        raise NotImplementedError
        from scipy import weave
        N = y.shape[0]
        h = gh_w.shape[0]
        F = np.zeros((y.shape[0], D))
        #f_string = "pow(x(n,i), y(n,j))"

        support_code = """
        #include <stdio.h>
        """
        omp = False
        if omp:
            pragma = "#pragma omp parallel for private(d, n, i, j)"
            support_code += """
            #include <math.h>
            #include <omp.h>
            """
            weave_options = {'headers'           : ['<omp.h>'],
                            'extra_compile_args': ['-fopenmp -O3'], # -march=native'],
                            'extra_link_args'   : ['-lgomp']}
        else:
            pragma = ""
            weave_options = {}


        code = """
        int d,n,i,j;
        {pragma}
        for(d=0; d<D; d++){{
            for(n=0; n<N; n++){{
                for(i=0; i<h; i++){{
                    for(j=0; j<h; j++){{
                        F(n,d) += gh_w(i)*gh_w(j)*{func};
                    }}
                }}
            }}
        }}
        """.format(func=f_string, pragma=pragma)

        weave.inline(code, ['F', 'f', 'g', 'y', 'v', 'N', 'D', 'h', 'gh_w', 'e_g'],
                     type_converters=weave.converters.blitz,
                     support_code=support_code, **weave_options)
        return F

