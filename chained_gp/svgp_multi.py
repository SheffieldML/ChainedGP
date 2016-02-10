# Copyright (c) 2014, James Hensman, Alex Matthews
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
from GPy.util import choleskies
from GPy.core.model import Model
from GPy.core.parameterization.param import Param
from svgp_multi_inf import SVGPMultiInf as svgp_inf
from GPy.util.linalg import mdot
from GPy.core.parameterization.variational import VariationalPosterior, NormalPosterior
from GPy.core.parameterization import ObsAr
import GPy
import sys
from GPy.plotting.matplot_dep.util import fixed_inputs
import matplotlib.pyplot as plt

class SVGPMulti(GPy.core.SparseGP):
    def __init__(self, X, Y, Z, kern_list, likelihood, mean_functions=None, name='SVGPMulti', Y_metadata=None, batchsize=None):
        """
        Extension to the SVGP to allow multiple latent function,
        where the latent functions are assumed independant (have one kernel per latent function)
        """
        # super(SVGPMulti, self).__init__(name)  # Parameterized.__init__(self)

        assert X.ndim == 2
        self.Y_metadata = Y_metadata
        _, self.output_dim = Y.shape

        # self.Z = Param('inducing inputs', Z)
        # self.num_inducing = Z.shape[0]
        # self.likelihood = likelihood

        self.kern_list = kern_list
        self.batchsize = batchsize

        #Batch the data
        self.X_all, self.Y_all = X, Y
        if batchsize is None:
            X_batch, Y_batch = X, Y
        else:
            import climin.util
            #Make a climin slicer to make drawing minibatches much quicker
            self.slicer = climin.util.draw_mini_slices(self.X_all.shape[0], self.batchsize)
            X_batch, Y_batch = self.new_batch()

        # if isinstance(X_batch, (ObsAr, VariationalPosterior)):
            # self.X = X_batch.copy()
        # else:
            # self.X = ObsAr(X_batch)
        # self.Y = Y_batch

        #create the SVI inference method
        # self.inference_method = svgp_inf()
        inference_method = svgp_inf()

        #Initialize base model
        super(SVGPMulti, self).__init__(X=X_batch, Y=Y_batch, Z=Z, kernel=kern_list[0], likelihood=likelihood, mean_function=None, X_variance=None, inference_method=inference_method, name=name, Y_metadata=Y_metadata, normalizer=False)
        self.unlink_parameter(self.kern)  # We don't want a single kern

        # self.num_data, self.input_dim = self.X.shape
        self.num_outputs = self.Y.shape[1]

        self.num_latent_funcs = self.likelihood.request_num_latent_functions(self.Y_all)

        #Make a latent function per dimension
        self.q_u_means = Param('q_u_means', np.zeros((self.num_inducing, self.num_latent_funcs)))
        chols = choleskies.triang_to_flat(np.tile(np.eye(self.num_inducing)[None,:,:], (self.num_latent_funcs,1,1)))
        self.q_u_chols = Param('qf_u_chols', chols)

        self.link_parameter(self.Z, index=0)
        self.link_parameter(self.q_u_means)
        self.link_parameter(self.q_u_chols)
        # self.link_parameter(self.likelihood)

        #Must pass a list of kernels that work on each latent function for now
        assert len(kern_list) == self.num_latent_funcs
        #Add the rest of the kernels, one kernel per latent function
        [self.link_parameter(kern) for kern in kern_list]
        #self.latent_f_list = [self.mf, self.mg]
        #self.latent_fchol_list = [self.cholf, self.cholg]

        if mean_functions is None:
            self.mean_functions = [None]*self.num_latent_funcs
        elif len(mean_functions) != len(kern_list):
            raise ValueError("Must provide a mean function for all latent\n\
                             functions as a list, provide None if no latent\n\
                             function is needed for a specific latent function")
        else:
            self.mean_functions = []
            for m_f in mean_functions:
                if m_f is not None:
                    self.link_parameter(m_f)
                self.mean_functions.append(m_f)


    def log_likelihood(self):
        return self._log_marginal_likelihood

    def parameters_changed(self):
        self.batch_scale = float(self.X_all.shape[0])/float(self.X.shape[0])
        self.posteriors, self._log_marginal_likelihood, grad_dict = self.inference_method.inference(self.q_u_means, self.q_u_chols, self.kern_list, self.X, self.Z, self.likelihood,
                                                                                                        self.Y, self.mean_functions, self.Y_metadata, KL_scale=1.0, batch_scale=self.batch_scale)
        self.likelihood.update_gradients(grad_dict['dL_dthetaL'])
        #update the kernel gradients
        #Shared Z
        Z_grad = np.zeros_like(self.Z.values)
        for latent_f_ind, kern in enumerate(self.kern_list):
            kern.update_gradients_full(grad_dict['dL_dKmm'][latent_f_ind], self.Z)
            grad = kern.gradient.copy()
            kern.update_gradients_full(grad_dict['dL_dKmn'][latent_f_ind], self.Z, self.X)
            grad += kern.gradient.copy()
            kern.update_gradients_diag(grad_dict['dL_dKdiag'][latent_f_ind], self.X)
            kern.gradient += grad
            if not self.Z.is_fixed:# only compute these expensive gradients if we need them
                Z_grad += kern.gradients_X(grad_dict['dL_dKmm'][latent_f_ind], self.Z)
                Z_grad += kern.gradients_X(grad_dict['dL_dKmn'][latent_f_ind], self.Z, self.X)

            #update the variational parameter gradients:
            self.q_u_means[:, latent_f_ind*self.num_outputs:(latent_f_ind+1)*self.num_outputs].gradient = grad_dict['dL_dm'][latent_f_ind]
            self.q_u_chols[:, latent_f_ind*self.num_outputs:(latent_f_ind+1)*self.num_outputs].gradient = grad_dict['dL_dchol'][latent_f_ind]

            mean_function = self.mean_functions[latent_f_ind]
            if mean_function is not None:
                mean_function.update_gradients(grad_dict['dL_dmfX'][latent_f_ind], self.X)
                g = mean_function.gradient[:].copy()
                mean_function.update_gradients(grad_dict['dL_dmfZ'][latent_f_ind], self.Z)
                mean_function.gradient[:] += g
                Z_grad += mean_function.gradients_X(grad_dict['dL_dmfZ'][latent_f_ind], self.Z)

        if not self.Z.is_fixed:# only compute these expensive gradients if we need them
            self.Z.gradient[:] = Z_grad

    def set_data(self, X, Y):
        """
        Set the data without calling parameters_changed to avoid wasted computation
        If this is called by the stochastic_grad function this will immediately update the gradients
        """
        assert X.shape[1]==self.Z.shape[1]
        self.X, self.Y = X, Y

    def new_batch(self):
        """
        Return a new batch of X and Y by taking a chunk of data from the complete X and Y
        """
        i = self.slicer.next()
        return self.X_all[i], self.Y_all[i]

    def stochastic_grad(self, parameters):
        self.set_data(*self.new_batch())
        return self._grads(parameters)

    def optimizeWithFreezingZ(self):
        self.Z.fix()
        self.kern.fix()
        self.optimize('bfgs')
        self.Z.unfix()
        self.kern.constrain_positive()
        self.optimize('bfgs')

    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        mf, vf = self._raw_predict(x_test, 0)
        mg, vg = self._raw_predict(x_test, 1)
        mu_stars = [mf, mg]
        var_stars = [vf, vg]
        return self.likelihood.log_predictive_density(y_test, mu_stars, var_stars, Y_metadata)

    def log_predictive_density_sampling(self, x_test, y_test, Y_metadata=None, num_samples=1000):
        mf, vf = self._raw_predict(x_test, 0)
        mg, vg = self._raw_predict(x_test, 1)
        mu_stars = np.hstack((mf, mg))
        var_stars = np.hstack((vf, vg))
        return self.likelihood.log_predictive_density_sampling(y_test, mu_stars, var_stars, Y_metadata, num_samples=num_samples)

    def _raw_predict(self, Xnew, latent_function_ind=None, full_cov=False, kern=None):
        """
        Make a prediction for the latent function values.

        For certain inputs we give back a full_cov of shape NxN,
        if there is missing data, each dimension has its own full_cov of shape NxNxD, and if full_cov is of,
        we take only the diagonal elements across N.

        For uncertain inputs, the SparseGP bound produces a full covariance structure across D, so for full_cov we
        return a NxDxD matrix and in the not full_cov case, we return the diagonal elements across D (NxD).
        This is for both with and without missing data. See for missing data SparseGP implementation py:class:'~GPy.models.sparse_gp_minibatch.SparseGPMiniBatch'.
        """
        #Plot f by default
        if latent_function_ind is None:
            latent_function_ind = 0

        if kern is None:
            kern = self.kern_list[latent_function_ind]

        posterior = self.posteriors[latent_function_ind]

        Kx = kern.K(self.Z, Xnew)
        mu = np.dot(Kx.T, posterior.woodbury_vector)
        if full_cov:
            Kxx = kern.K(Xnew)
            if posterior.woodbury_inv.ndim == 2:
                var = Kxx - np.dot(Kx.T, np.dot(posterior.woodbury_inv, Kx))
            elif posterior.woodbury_inv.ndim == 3:
                var = Kxx[:,:,None] - np.tensordot(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx).T, Kx, [1,0]).swapaxes(1,2)
            var = var
        else:
            Kxx = kern.Kdiag(Xnew)
            var = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T
        #add in the mean function
        if self.mean_functions[latent_function_ind] is not None:
            mu += self.mean_functions[latent_function_ind].f(Xnew)

        return mu, var

    def plot_fs(self, dim=0, variances=False, median=True, true_variance=True):
        """
        Plotting for models with two latent functions, one is an exponent over the scale
        parameter
        """
        assert self.likelihood.request_num_latent_functions(self.Y) == 2
        if median:
            XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='median', as_list=False)
        else:
            XX = np.linspace(self.X[:, dim].min(), self.X[:, dim].max(), 200)[:, None]
        X_pred_points = XX.copy()
        X_pred_points_lin = np.linspace(self.X[:, dim].min(), self.X[:, dim].max(), self.X.shape[0])
        X_pred_points[:, dim] = X_pred_points_lin

        mf, vf = self._raw_predict(X_pred_points, 0)
        mg, vg = self._raw_predict(X_pred_points, 1)

        f_std = np.sqrt(vf)
        mf_lower = mf - 2*f_std
        mf_upper = mf + 2*f_std

        if true_variance:
            #Real likelihood variance
            g_std = np.sqrt(self.likelihood.conditional_variance(mg))
            g_std_err_f = 2*np.sqrt(np.exp(vg))  # Standard error in f space
            vg_std = np.sqrt(self.likelihood.conditional_variance(g_std_err_f))  # std error in likelihood space
        else:
            #Squared scale parameter
            g_std = np.sqrt(np.exp(mg))
            vg_std = np.sqrt(vg)

        mg_loc_upper = mf + 2*g_std
        mg_loc_lower = mf - 2*g_std

        fig, ax = plt.subplots()
        X_dim = X_pred_points[:,dim:dim+1]
        ax.plot(X_dim, mf, 'b-')
        ax.plot(X_dim, mg_loc_upper, 'g-')
        ax.plot(X_dim, mg_loc_lower, 'm-')
        ax.plot(XX, self.Y, 'kx')

        if variances:
            ax.plot(X_dim, mf_upper, 'b--', alpha=0.5)
            ax.plot(X_dim, mf_lower, 'b--', alpha=0.5)

            gf_upper_upper = mg_loc_upper + 2*vg_std
            gf_upper_lower = mg_loc_upper - 2*vg_std

            gf_lower_upper = mg_loc_lower + 2*vg_std
            gf_lower_lower = mg_loc_lower - 2*vg_std

            #Variance around upper standard erro
            ax.plot(X_dim, gf_upper_upper, 'g--', alpha=0.5)
            ax.plot(X_dim, gf_upper_lower, 'g--', alpha=0.5)

            #Variance around lower standard error
            ax.plot(X_dim, gf_lower_upper, 'm--', alpha=0.5)
            ax.plot(X_dim, gf_lower_lower, 'm--', alpha=0.5)

    # def plot_f(self, plot_limits=None, which_data_rows='all',
        # which_data_ycols='all', fixed_inputs=[],
        # levels=20, samples=0, fignum=None, ax=None, resolution=None,
        # plot_raw=True,
        # linecol=None,fillcol=None, Y_metadata=None, data_symbol='kx',
        # apply_link=False):
        # """
        # Plot the GP's view of the world, where the data is normalized and before applying a likelihood.
        # This is a call to plot with plot_raw=True.
        # Data will not be plotted in this, as the GP's view of the world
        # may live in another space, or units then the data.

        # Can plot only part of the data and part of the posterior functions
        # using which_data_rowsm which_data_ycols.

        # :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        # :type plot_limits: np.array
        # :param which_data_rows: which of the training data to plot (default all)
        # :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        # :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        # :type which_data_ycols: 'all' or a list of integers
        # :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        # :type fixed_inputs: a list of tuples
        # :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        # :type resolution: int
        # :param levels: number of levels to plot in a contour plot.
        # :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        # :type levels: int
        # :param samples: the number of a posteriori samples to plot
        # :type samples: int
        # :param fignum: figure to plot on.
        # :type fignum: figure number
        # :param ax: axes to plot on.
        # :type ax: axes handle
        # :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        # :type linecol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        # :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        # :type fillcol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        # :param Y_metadata: additional data associated with Y which may be needed
        # :type Y_metadata: dict
        # :param data_symbol: symbol as used matplotlib, by default this is a black cross ('kx')
        # :type data_symbol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) alongside marker type, as is standard in matplotlib.
        # :param apply_link: if there is a link function of the likelihood, plot the link(f*) rather than f*
        # :type apply_link: boolean
        # """
        # assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        # from GPy.plotting.matplot_dep import models_plots
        # kw = {}
        # if linecol is not None:
            # kw['linecol'] = linecol
        # if fillcol is not None:
            # kw['fillcol'] = fillcol
        # return models_plots.plot_fit(self, plot_limits, which_data_rows,
                                     # which_data_ycols, fixed_inputs,
                                     # levels, samples, fignum, ax, resolution,
                                     # plot_raw=plot_raw, Y_metadata=Y_metadata,
                                     # data_symbol=data_symbol, apply_link=apply_link, **kw)

    # def plot(self, plot_limits=None, which_data_rows='all',
        # which_data_ycols='all', fixed_inputs=[],
        # levels=20, samples=0, fignum=None, ax=None, resolution=None,
        # plot_raw=False,
        # linecol=None,fillcol=None, Y_metadata=None, data_symbol='kx'):
        # """
        # Plot the posterior of the GP.
          # - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          # - In two dimsensions, a contour-plot shows the mean predicted function
          # - In higher dimensions, use fixed_inputs to plot the GP  with some of the inputs fixed.

        # Can plot only part of the data and part of the posterior functions
        # using which_data_rowsm which_data_ycols.

        # :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        # :type plot_limits: np.array
        # :param which_data_rows: which of the training data to plot (default all)
        # :type which_data_rows: 'all' or a slice object to slice model.X, model.Y
        # :param which_data_ycols: when the data has several columns (independant outputs), only plot these
        # :type which_data_ycols: 'all' or a list of integers
        # :param fixed_inputs: a list of tuple [(i,v), (i,v)...], specifying that input index i should be set to value v.
        # :type fixed_inputs: a list of tuples
        # :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        # :type resolution: int
        # :param levels: number of levels to plot in a contour plot.
        # :param levels: for 2D plotting, the number of contour levels to use is ax is None, create a new figure
        # :type levels: int
        # :param samples: the number of a posteriori samples to plot
        # :type samples: int
        # :param fignum: figure to plot on.
        # :type fignum: figure number
        # :param ax: axes to plot on.
        # :type ax: axes handle
        # :param linecol: color of line to plot [Tango.colorsHex['darkBlue']]
        # :type linecol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        # :param fillcol: color of fill [Tango.colorsHex['lightBlue']]
        # :type fillcol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) as is standard in matplotlib
        # :param Y_metadata: additional data associated with Y which may be needed
        # :type Y_metadata: dict
        # :param data_symbol: symbol as used matplotlib, by default this is a black cross ('kx')
        # :type data_symbol: color either as Tango.colorsHex object or character ('r' is red, 'g' is green) alongside marker type, as is standard in matplotlib.
        # """
        # assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        # from GPy.plotting.matplot_dep import models_plots
        # kw = {}
        # if linecol is not None:
            # kw['linecol'] = linecol
        # if fillcol is not None:
            # kw['fillcol'] = fillcol
        # return models_plots.plot_fit(self, plot_limits, which_data_rows,
                                     # which_data_ycols, fixed_inputs,
                                     # levels, samples, fignum, ax, resolution,
                                     # plot_raw=plot_raw, Y_metadata=Y_metadata,
                                     # data_symbol=data_symbol, **kw)
