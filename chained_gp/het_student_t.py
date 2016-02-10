# Copyright (c) 2012-2014 Ricardo Andrade, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from scipy.special import gammaln, gamma
import scipy as sp
#from scipy import stats, integrate
from GPy.likelihoods import link_functions
#from GPy.likelihoods.likelihood import Likelihood
from multi_likelihood import MultiLikelihood
from GPy.core.parameterization import Param
from scipy.special import psi as digamma
from scipy.special import gamma
from GPy.core.parameterization.transformations import Logexp
import quad_cython
from functools import partial

class HetStudentT(MultiLikelihood):
    """
    Student T likelihood

    For nomanclature see Bayesian Data Analysis 2003 p576

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\frac{\\Gamma\\left(\\frac{v+1}{2}\\right)}{\\Gamma\\left(\\frac{v}{2}\\right)\\sqrt{v\\pi\\exp(g_{i})}}\\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - f_{i})^{2}}{\\exp(g_{i})}\\right)\\right)^{\\frac{-v+1}{2}}

    """
    def __init__(self,gp_link=None, deg_free=5, sigma2=2):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(HetStudentT, self).__init__(gp_link, name='Hetro_Student_T')
        self.v = Param('deg_free', float(deg_free), Logexp())
        self.link_parameter(self.v)
        self.v.constrain_fixed()

        self.log_concave = False

    def request_num_latent_functions(self, Y):
        """
        The likelihood should infer how many latent functions are needed for the likelihood

        In this case we have one latent function for mean and one for scale, for each output dimension
        """
        return Y.shape[1]*2

    def pdf(self, f, g, y, Y_metadata=None):
        """
        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\frac{\\Gamma\\left(\\frac{v+1}{2}\\right)}{\\Gamma\\left(\\frac{v}{2}\\right)\\sqrt{v\\pi\\exp(g_{i})}}\\left(1 + \\frac{1}{v}\\left(\\frac{(y_{i} - f_{i})^{2}}{\\exp(g_{i})}\\right)\\right)^{\\frac{-v+1}{2}}
        """
        df = float(self.deg_free[:])
        e_g = np.exp(g)
        y_f2 = (y-f)**2
        pdf = (gamma(0.5*(df+1)) / (gamma(0.5*df)*np.sqrt(df*np.pi*e_g)))*(1 + y_f2/(df*e_g))**(-0.5*(df+1))
        return pdf

    def logpdf(self, f, y, Y_metadata=None):
        D = y.shape[1]
        fv, gv = f[:, :D], f[:, D:]
        df = float(self.deg_free[:])
        y_f2 = (y-fv)**2
        lnpdf = gammaln(0.5*(df+1)) - gammaln(0.5*df) - 0.5*np.log(df*np.pi) - 0.5*gv - 0.5*(df+1)*np.log1p(y_f2/(df*np.exp(gv)))
        return lnpdf

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        self.v.gradient = grads[0]

    def predictive_mean(self, mu, sigma, Y_metadata=None):
        # The comment here confuses mean and median.
        return self.gp_link.transf(mu) # only true if link is monotonic, which it is.

    def predictive_variance(self, mu,variance, predictive_mean=None, Y_metadata=None):
        if self.deg_free<=2.:
            return np.empty(mu.shape)*np.nan # does not exist for degrees of freedom <= 2.
        else:
            return super(StudentT, self).predictive_variance(mu, variance, predictive_mean, Y_metadata)

    def conditional_mean(self, gp):
        return self.gp_link.transf(gp)

    def conditional_variance(self, gp):
        #Expects just g!
        return self.deg_free*np.exp(gp)/(self.deg_free - 2.)

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        #FIXME: Very slow as we are computing a new random variable per input!
        #Can't get it to sample all at the same time
        #student_t_samples = np.array([stats.t.rvs(self.v, self.gp_link.transf(gpj),scale=np.sqrt(self.sigma2), size=1) for gpj in gp])
        dfs = np.ones_like(gp)*self.v
        scales = np.ones_like(gp)*np.sqrt(self.sigma2)
        student_t_samples = stats.t.rvs(dfs, loc=self.gp_link.transf(gp),
                                        scale=scales)
        return student_t_samples.reshape(orig_shape)

    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        df = float(self.deg_free[:])
        #F =  -0.5*mg
        #Parameterize sigma not sigma2 as sigma itself needs to be positive!
        #F = -mg

        F = (gammaln((df + 1) * 0.5)
            - gammaln(df * 0.5)
            - 0.5*np.log(df * np.pi * np.exp(mg))
             )

        """
        #Some little code to check the result numerically using quadrature
        from functools import partial
        from scipy import integrate
        i = 5  # datapoint index
        def quad_func(fi, gi, yi, mgi, vgi, mfi, vfi):
            return ((-0.5*(df+1)*np.log1p(((yi-fi)**2)/(df*np.exp(gi))))       #p(y|f,g)
                    * np.exp(-0.5*np.log(2*np.pi*vgi) - 0.5*((gi - mgi)**2)/vgi) #q(g)
                    * np.exp(-0.5*np.log(2*np.pi*vfi) - 0.5*((fi - mfi)**2)/vfi) #q(f)
                    )
        quad_func_l = partial(quad_func, yi=Y[i], mgi=mg[i], vgi=vg[i], mfi=mf[i], vfi=vf[i])
        def integrl(gi):
            return integrate.quad(quad_func_l, -50, 50, args=(gi))[0]
        print "Numeric scipy F quad"
        print integrate.quad(lambda fi: integrl(fi), -50, 50)
        """

        #Do some testing to see if the quadrature works well for one datapoint
        Ngh = 20
        if gh_points is None:
            gh_x, gh_w = self._gh_points(T=Ngh)
        else:
            gh_x, gh_w = gh_points

        N = Y.shape[0]
        F_quad = np.zeros(Y.shape)
        dF_dmg = np.zeros(mg.shape)
        dF_dmf = np.zeros(mf.shape)
        dF_dvf = np.zeros(vf.shape)
        dF_dvg = np.zeros(vg.shape)
        dF_ddf = np.zeros(vg.shape)
        for d in range(D):
            quad_cython.quad2d_stut(N, mf.flatten(), vf.flatten(), mg.flatten(), vg.flatten(), Y.flatten(), Ngh, df,
                                    gh_x, gh_w, F_quad[:,d], dF_dmf[:,d], dF_dvf[:,d], dF_dmg[:,d], dF_dvg[:,d], dF_ddf[:,d])
        F_quad /= np.pi
        dF_dmg /= np.pi
        dF_dmf /= np.pi
        dF_dvf /= np.pi
        dF_dvg /= np.pi
        dF_ddf /= np.pi

        F += F_quad
        dF_dmg += -0.5  # from -0.5<g> term

        dF_dvf /= 2.0
        dF_dvg /= 2.0

        dF_dm = np.hstack((dF_dmf, dF_dmg))
        dF_dv = np.hstack((dF_dvf, dF_dvg))

        #derivative wrt to degrees of freedom
        dF_ddf += 0.5*digamma(0.5*(df+1)) - 0.5*digamma(0.5*df) - 1.0/(2*df)
        #Since we are the only parameter, our first dimension is 1
        dF_dtheta = dF_ddf[None, :]
        return F, dF_dm, dF_dv, dF_dtheta

    def variational_expectations_pure(self, Y, m, v, gh_points=None, Y_metadata=None):
        D = Y.shape[1]
        mf, mg = m[:, :D], m[:, D:]
        vf, vg = v[:, :D], v[:, D:]

        df = float(self.deg_free[:])
        #F =  -0.5*mg
        #Parameterize sigma not sigma2 as sigma itself needs to be positive!
        #F = -mg

        F = (gammaln((df + 1) * 0.5)
            - gammaln(df * 0.5)
            - 0.5*np.log(df * np.pi * np.exp(mg))
             )

        """
        #Some little code to check the result numerically using quadrature
        from functools import partial
        from scipy import integrate
        i = 5  # datapoint index
        def quad_func(fi, gi, yi, mgi, vgi, mfi, vfi):
            return ((-0.5*(df+1)*np.log1p(((yi-fi)**2)/(df*np.exp(gi))))       #p(y|f,g)
                    * np.exp(-0.5*np.log(2*np.pi*vgi) - 0.5*((gi - mgi)**2)/vgi) #q(g)
                    * np.exp(-0.5*np.log(2*np.pi*vfi) - 0.5*((fi - mfi)**2)/vfi) #q(f)
                    )
        quad_func_l = partial(quad_func, yi=Y[i], mgi=mg[i], vgi=vg[i], mfi=mf[i], vfi=vf[i])
        def integrl(gi):
            return integrate.quad(quad_func_l, -50, 50, args=(gi))[0]
        print "Numeric scipy F quad"
        print integrate.quad(lambda fi: integrl(fi), -50, 50)
        """

        from functools import partial
        def F_quad_func(f, e_g, y, df):
            y_f2 = (y-f)**2
            return -0.5*(df+1)*np.log1p(y_f2/(df*e_g))

        def F_dquad_df_func(f, e_g, y, df):
            y_f2 = (y-f)**2
            return (df+1)*(y-f)/(df*e_g + y_f2)

        def F_d2quad_df2_func(f, e_g, y, df):
            y_f2 = (y-f)**2
            df_eg = df*e_g
            return (df+1)*(y_f2 - df_eg)/(df_eg + y_f2)**2

        def F_dquad_dg_func(f, e_g, y, df):
            y_f2 = (y-f)**2
            return 0.5*(df+1)*y_f2/(df*e_g + y_f2)

        def F_d2quad_dg2_func(f, e_g, y, df):
            y_f2 = (y-f)**2
            df_eg = df*e_g
            return -0.5*(df+1)*y_f2*df_eg/(df_eg + y_f2)**2

        def F_dquad_ddf_func(f, e_g, y, df):
            y_f2 = (y-f)**2
            df_eg = df*e_g
            return 0.5*( (df+1)*y_f2/(df*(df_eg + y_f2))
                        - np.log1p(y_f2/(df_eg))
                        )

        F_quad_func_p = partial(F_quad_func, df=df)
        F_dquad_df_func_p = partial(F_dquad_df_func, df=df)
        F_d2quad_df2_func_p = partial(F_d2quad_df2_func, df=df)
        F_dquad_dg_func_p = partial(F_dquad_dg_func, df=df)
        F_d2quad_dg2_func_p = partial(F_d2quad_dg2_func, df=df)
        F_dquad_ddf_func_p = partial(F_dquad_ddf_func, df=df)

        #(F_quad, dF_dmf, dF_dvf,
        #dF_dmg, dF_dvg) = self.quad2d([F_quad_func_p, F_dquad_df_func_p,
                                               #F_d2quad_df2_func_p, F_dquad_dg_func_p,
                                               #F_d2quad_dg2_func_p],#, F_dquad_ddf_func_p],
                                              #Y, mf, vf, mg, vg, gh_points, exp_g=True)
        (F_quad, dF_dmf, dF_dvf,
        dF_dmg, dF_dvg, dF_ddf) = self.quad2d([F_quad_func_p, F_dquad_df_func_p,
                                               F_d2quad_df2_func_p, F_dquad_dg_func_p,
                                               F_d2quad_dg2_func_p, F_dquad_ddf_func_p],
                                              Y, mf, vf, mg, vg, gh_points, exp_g=True)

        F += F_quad
        dF_dmg += -0.5  # from -0.5<g> term

        dF_dvf /= 2.0
        dF_dvg /= 2.0

        dF_dm = np.hstack((dF_dmf, dF_dmg))
        dF_dv = np.hstack((dF_dvf, dF_dvg))

        #derivative wrt to degrees of freedom
        dF_ddf += 0.5*digamma(0.5*(df+1)) - 0.5*digamma(0.5*df) - 1.0/(2*df)
        #Since we are the only parameter, our first dimension is 1
        dF_dtheta = dF_ddf[None, :]
        return F, dF_dm, dF_dv, dF_dtheta

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
