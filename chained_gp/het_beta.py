# Copyright (c) 2015 Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import beta, betaln, gammaln, gamma, polygamma, zeta
from scipy.special import psi
#from scipy import stats, integrate
from GPy.likelihoods import link_functions
#from GPy.likelihoods.likelihood import Likelihood
from multi_likelihood import MultiLikelihood
#import quad_cython
from functools import partial
def approx_2zeta(n):
   """
   http://math.stackexchange.com/questions/914416/hint-on-a-limit-that-involves-the-hurwitz-zeta-function

   For Hurwitz zeta function when q=2 the taylor expansion is a good approximation

   Appears only for large n
   """
   return 1.0/n + 1.0/(2*n**2) + 1.0/(6*n**3) - 1.0/(30*n**5)

def approx_2zeta_sum(n):
    """
    Zeta is just a sum up to infinite, just cap it at 100
    """
    #Only works with 2d arrays
    assert len(n.shape) == 2
    k = np.arange(30)[:, None, None]
    return np.sum(1.0/(k + n)**2, axis=0)

class HetBeta(MultiLikelihood):
    """
    Beta likelihood with latent functions over both alpha and beta

    """
    def __init__(self,gp_link=None, deg_free=5, sigma2=2):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(HetBeta, self).__init__(gp_link, name='Hetra_beta')
        self.log_concave = False

    def request_num_latent_functions(self, Y):
        """
        The likelihood should infer how many latent functions are needed for the likelihood

        In this case we have one latent function for both shape parameters, for each output dimension
        """
        return Y.shape[1]*2

    def pdf(self, f, g, y, Y_metadata=None):
        ef = np.exp(f)
        eg = np.exp(g)
        pdf = y**(ef-1) * (1-y)**(eg-1) / beta(ef, eg)
        return pdf

    def logpdf(self, f, y, Y_metadata=None):
        D = y.shape[1]
        fv, gv = f[:, :D], f[:, D:]
        ef = np.exp(fv)
        eg = np.exp(gv)
        lnpdf = (ef - 1)*np.log(y) + (eg - 1)*np.log(1-y) - betaln(ef, eg)
        return lnpdf

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
        orig_shape = gp.shape
        gp = gp.flatten()
        #FIXME: Very slow as we are computing a new random variable per input!
        #Can't get it to sample all at the same time
        #beta_samples = np.array([stats.t.rvs(self.v, self.gp_link.transf(gpj),scale=np.sqrt(self.sigma2), size=1) for gpj in gp])
        #scales = np.ones_like(gp)*np.sqrt(self.sigma2)
        #beta_samples = stats.t.rvs(dfs, loc=self.gp_link.transf(gp),
                                        #scale=scales)
        return beta_samples.reshape(orig_shape)

    def variational_expectations_cython(self, Y, m, v, gh_points=None, Y_metadata=None):
        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        lnY = np.log(Y)
        neglnY = np.log(Y)
        F = (np.exp(mf + .5*vf) - 1)*lnY + (np.exp(mg + .5*vg) - 1)*neglnY
        #Some little code to check the result numerically using quadrature
        from scipy import integrate
        i = 5  # datapoint index
        def quad_func(fi, gi, yi, mgi, vgi, mfi, vfi):
            return ((-gammaln(np.exp(fi)) - gammaln(np.exp(gi)) + gammaln(np.exp(fi) + np.exp(gi)))      #p(y|f,g)
                    * np.exp(-0.5*np.log(2*np.pi*vgi) - 0.5*((gi - mgi)**2)/vgi) #q(g)
                    * np.exp(-0.5*np.log(2*np.pi*vfi) - 0.5*((fi - mfi)**2)/vfi) #q(f)
                    )
        quad_func_l = partial(quad_func, yi=Y[i], mgi=mg[i], vgi=vg[i], mfi=mf[i], vfi=vf[i])
        def integrl(gi):
            return integrate.quad(quad_func_l, -50, 50, args=(gi))[0]
        print "Numeric scipy F quad"
        print integrate.quad(lambda fi: integrl(fi), -50, 50)

        #Do some testing to see if the quadrature works well for one datapoint
        Ngh = 20
        if gh_points is None:
            gh_x, gh_w = self._gh_points(T=Ngh)
        else:
            gh_x, gh_w = gh_points

        F_quad = np.zeros(Y.shape)
        dF_dmg = np.zeros(mg.shape)
        dF_dmf = np.zeros(mf.shape)
        dF_dvf = np.zeros(vf.shape)
        dF_dvg = np.zeros(vg.shape)
        dF_ddf = np.zeros(vg.shape)
        quad_cython.quad2d_beta(mf, vf, mg, vg, Y, gh_x, gh_w, F_quad, dF_dmf, dF_dvf, dF_dmg, dF_dvg)

        F_quad /= np.pi
        dF_dmg /= np.pi
        dF_dmf /= np.pi
        dF_dvf /= np.pi
        dF_dvg /= np.pi
        dF_ddf /= np.pi

        F += F_quad

        dF_dvf += 0.0 # ?
        dF_dmf += 0.0 # ?
        dF_dmg += 0.0 # ?
        dF_dvg += 0.0 # ?

        dF_dvf /= 2.0
        dF_dvg /= 2.0

        dF_dm = np.hstack((dF_dmf, dF_dmg))
        dF_dv = np.hstack((dF_dvf, dF_dvg))

        #Since we are the only parameter, our first dimension is 1
        return F, dF_dm, dF_dv, None

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        from GPy.util.misc import safe_exp, safe_square

        lnY = np.log(Y)
        neglnY = np.log(1.0-Y)
        F = (safe_exp(mf + .5*vf) - 1)*lnY + (safe_exp(mg + .5*vg) - 1)*neglnY

        ##Some little code to check the result numerically using quadrature
        #from scipy import integrate
        #i = 6  # datapoint index
        #def quad_func(fi, gi, yi, mgi, vgi, mfi, vfi):
            #return ((-gammaln(np.exp(fi)) - gammaln(np.exp(gi)) + gammaln(np.exp(fi) + np.exp(gi)))      #p(y|f,g)
                    #* np.exp(-0.5*np.log(2*np.pi*vgi) - 0.5*((gi - mgi)**2)/vgi) #q(g)
                    #* np.exp(-0.5*np.log(2*np.pi*vfi) - 0.5*((fi - mfi)**2)/vfi) #q(f)
                    #)
        #quad_func_l = partial(quad_func, yi=Y[i], mgi=mg[i], vgi=vg[i], mfi=mf[i], vfi=vf[i])
        #def integrl(gi):
            #return integrate.quad(quad_func_l, -50, 50, args=(gi))[0]
        #print "Numeric scipy F quad"
        #print integrate.quad(lambda fi: integrl(fi), -50, 50)

        def F_quad_func(e_f, e_g, y):
            return -gammaln(e_f) - gammaln(e_g) + gammaln(e_f + e_g)

        def F_dquad_df_func(e_f, e_g, y):
            #return -polygamma(0, e_f)*e_f + polygamma(0, e_f + e_g)*e_f
            return -psi(e_f)*e_f + psi(e_f + e_g)*e_f

        def F_d2quad_df2_func(e_f, e_g, y):
            e_2f = safe_square(e_f)
            #return 0.5*(-polygamma(1, e_f)*e_2f - polygamma(0, e_f)*e_f
                        #+polygamma(1, e_f + e_g)*e_2f + polygamma(0, e_f + e_g)*e_f)
            return 0.5*(-zeta(2, e_f)*e_2f - psi(e_f)*e_f
                        +zeta(2, e_f + e_g)*e_2f + psi(e_f + e_g)*e_f)

        def F_d2quad_df2_func_approx(e_f, e_g, y):
            e_2f = safe_square(e_f)
            #return 0.5*(-polygamma(1, e_f)*e_2f - polygamma(0, e_f)*e_f
                        #+polygamma(1, e_f + e_g)*e_2f + polygamma(0, e_f + e_g)*e_f)
            #return 0.5*(-zeta(2, e_f)*e_2f - psi(e_f)*e_f
                        #+zeta(2, e_f + e_g)*e_2f + psi(e_f + e_g)*e_f)
            return 0.5*(-approx_2zeta(e_f)*e_2f - psi(e_f)*e_f
                        +approx_2zeta(e_f + e_g)*e_2f + psi(e_f + e_g)*e_f)

        def F_d2quad_df2_func_approx_sum(e_f, e_g, y):
            e_2f = safe_square(e_f)
            return 0.5*(-approx_2zeta_sum(e_f)*e_2f - psi(e_f)*e_f
                        +approx_2zeta_sum(e_f + e_g)*e_2f + psi(e_f + e_g)*e_f)


        def F_dquad_dg_func(e_f, e_g, y):
            #return -polygamma(0, e_g)*e_g + polygamma(0, e_f + e_g)*e_g
            return -psi(e_g)*e_g + psi(e_f + e_g)*e_g

        def F_d2quad_dg2_func(e_f, e_g, y):
            e_2g = safe_square(e_g)
            #return 0.5*(-polygamma(1, e_g)*e_2g - polygamma(0, e_g)*e_g
                        #+polygamma(1, e_f + e_g)*e_2g + polygamma(0, e_f + e_g)*e_g)
            return 0.5*(-zeta(2, e_g)*e_2g - psi(e_g)*e_g
                        +zeta(2, e_f + e_g)*e_2g + psi(e_f + e_g)*e_g)

        def F_d2quad_dg2_func_approx(e_f, e_g, y):
            e_2g = safe_square(e_g)
            return 0.5*(-approx_2zeta(e_g)*e_2g - psi(e_g)*e_g
                        +approx_2zeta(e_f + e_g)*e_2g + psi(e_f + e_g)*e_g)

        def F_d2quad_dg2_func_approx_sum(e_f, e_g, y):
            e_2g = safe_square(e_g)
            return 0.5*(-approx_2zeta_sum(e_g)*e_2g - psi(e_g)*e_g
                        +approx_2zeta_sum(e_f + e_g)*e_2g + psi(e_f + e_g)*e_g)

        #(F_quad, dF_dmf, dF_dvf,
        #dF_dmg, dF_dvg) = self.quad2d([F_quad_func, F_dquad_df_func,
                                               #F_d2quad_df2_func, F_dquad_dg_func,
                                               #F_d2quad_dg2_func],
                                              #Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)
        F_quad = self.quad2d([F_quad_func], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        dF_dmf = self.quad2d([F_dquad_df_func], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        #dF_dvf = self.quad2d([F_d2quad_df2_func], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        dF_dmg = self.quad2d([F_dquad_dg_func], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        #dF_dvg = self.quad2d([F_d2quad_dg2_func], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        #dF_dvf_approx = self.quad2d([F_d2quad_df2_func_approx], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        #dF_dvg_approx = self.quad2d([F_d2quad_dg2_func_approx], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        dF_dvg = self.quad2d([F_d2quad_dg2_func_approx_sum], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]
        dF_dvf = self.quad2d([F_d2quad_df2_func_approx_sum], Y, mf, vf, mg, vg, gh_points, exp_f=True, exp_g=True)[0]

        #print "f"
        #print np.min(dF_dvf)
        #print np.max(dF_dvf)
        #print np.max(dF_dvf - dF_dvf_approx_sum)
        #print np.min(dF_dvf - dF_dvf_approx_sum)
        #print "g"
        #print np.min(dF_dvg)
        #print np.max(dF_dvg)
        #print np.max(dF_dvg - dF_dvg_approx_sum)
        #print np.min(dF_dvg - dF_dvg_approx_sum)

        F += F_quad
        dF_dmf += safe_exp(mf + .5*vf)*lnY
        dF_dmg += safe_exp(mg + .5*vg)*neglnY
        dF_dvf += 0.5*safe_exp(mf + .5*vf)*lnY
        dF_dvg += 0.5*safe_exp(mg + .5*vg)*neglnY

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
        #quad_cython.quad2d_beta(mf, vf, mg, vg, Y, gh_x, gh_w, F_quad_cython,
                                  #dF_dmf_cython, dF_dvf_cython, dF_dmg_cython,
                                  #dF_dvg_cython)

        #F_quad_cython /= np.pi
        #dF_dmg_cython /= np.pi
        #dF_dmf_cython /= np.pi
        #dF_dvf_cython /= np.pi
        #dF_dvg_cython /= np.pi
        #dF_ddf_cython /= np.pi


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
