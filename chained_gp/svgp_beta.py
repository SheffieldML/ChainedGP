# Copyright (c) 2014, James Hensman, Alex Matthews
# Distributed under the terms of the GNU General public License, see LICENSE.txt

import numpy as np
from GPy.plotting.matplot_dep.util import fixed_inputs
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from svgp_multi import SVGPMulti
import matplotlib.dates as mdates
from matplotlib.dates import num2date
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors

class SVGPBeta(SVGPMulti):
    def __init__(self, X, Y, Z, kern_list, likelihood, mean_functions=None,
                 name='SVGPBeta', Y_metadata=None, batchsize=None):
        """
        Extension to the SVGP to allow multiple latent function,
        where the latent functions are assumed independant (have one kernel per latent function)
        """
        super(SVGPBeta, self).__init__(X=X, Y=Y, Z=Z, kern_list=kern_list, likelihood=likelihood, mean_functions=mean_functions,
                 name=name, Y_metadata=Y_metadata, batchsize=batchsize)


    def plot_fs(self, dim=0, variances=False, median=True, true_variance=True,
                    y_alpha=0.3, cmap=plt.cm.YlOrRd, num_pred_points=200,
                    X_scale=1.0, X_offset=0.0, plot_dates=True):
        """
        Plotting for models with two latent functions, one is an exponent over the scale
        parameter
        """
        assert self.likelihood.request_num_latent_functions(self.Y) == 2
        if median:
            XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='median', as_list=False, X_all=True)
        else:
            XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='mean', as_list=False, X_all=True)
        #Now we have the other values fixed, remake the matrix to be the right shape
        XX = np.zeros((num_pred_points, self.X_all.shape[1]))
        for d in range(self.X_all.shape[1]):
            XX[:, d] = self.X_all[0, d]
        X_pred_points = XX.copy()
        X_pred_points_lin = np.linspace(self.X_all[:, dim].min(), self.X_all[:, dim].max(), XX.shape[0])
        X_pred_points[:, dim] = X_pred_points_lin

        mf, covf = self._raw_predict(X_pred_points, 0, full_cov=True)
        mg, covg = self._raw_predict(X_pred_points, 1, full_cov=True)

        covf = covf[:,:,0]
        covg = covg[:,:,0]

        num_samples = 60
        samples_f = np.random.multivariate_normal(mf.flatten(), covf, num_samples)
        samples_g = np.random.multivariate_normal(mg.flatten(), covg, num_samples)

        alpha = np.exp(samples_f)
        beta = np.exp(samples_g)

        num_y_pixels = 60
        #Want the top left pixel to be evaluated at 1
        line = np.linspace(1, 0, num_y_pixels)
        res = np.zeros((X_pred_points.shape[0], num_y_pixels))
        for j in range(X_pred_points.shape[0]):
            sf = alpha[:, j]  # Pick out the jth point along X axis
            sg = beta[:, j]
            for i in range(num_samples):
                # Pick out the sample and evaluate the pdf on a line between 0
                # and 1 with these alpha and beta values
                res[j, :] += beta_dist.pdf(line, sf[i], sg[i])
            res[j, :] /= num_samples

        vmax, vmin = res[np.isfinite(res)].max(), res[np.isfinite(res)].min()

        norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)

        X_all = self.X_all*X_scale + X_offset
        X_pred_points = X_pred_points*X_scale + X_offset
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=2.0)
        ax1.set_title('averaged pdf and data')
        im = ax1.imshow(res.T, origin='upper',
                        extent=[X_pred_points[:,dim].min(),X_pred_points[:,dim].max(), 0, 1],
                        aspect='auto', cmap=cmap, norm=norm)
        fig.colorbar(im, orientation='horizontal', pad=0.2)
        if plot_dates:
            #All others should follow suit since we sharex
            ax1.plot_date(X_all, self.Y_all, 'kx', alpha=y_alpha)
        else:
            ax1.plot(X_all, self.Y_all, 'kx', alpha=y_alpha)

        #For labels
        ax2.set_title('Posterior GP for Beta distributed variables')
        ax2.plot(X_pred_points, beta.T[:,0], 'b-', label='beta', alpha=3./num_samples)
        ax2.plot(X_pred_points, alpha.T[:,0], 'm-', label='alpha', alpha=3./num_samples)

        #For rest of samples
        ax2.plot(X_pred_points, beta.T[:,1:], 'b-', alpha=3./num_samples)
        ax2.plot(X_pred_points, alpha.T[:,1:], 'm-', alpha=3./num_samples)

        ax3.plot(X_pred_points, alpha.T / (alpha.T + beta.T), 'b-', alpha=3./num_samples)
        ax3.set_title('Mean')

        var = (alpha.T*beta.T) / ((alpha.T + beta.T)**2 * (alpha.T+beta.T +1))
        ax4.plot(X_pred_points, var, 'b-', alpha=3./num_samples)
        ax4.set_title('variance')

        for i in range(num_samples):
            a = alpha[i, :]
            b = beta[i, :]
            mode = (a - 1) / (a + b - 2)
            mode = np.where(mode < 0, np.nan, mode)
            ax5.plot(X_pred_points, mode, 'b-', alpha=3./num_samples)
        ax5.set_title('Modes where they exist (alpha > 1, beta > 1)')
        ax5.set_ylim(0,1)
        plt.legend()

        ax1.set_xlim(X_pred_points[:, dim].min(), X_pred_points[:, dim].max())

        fig3d = plt.figure(figsize=(13,5))
        ax = fig3d.add_subplot(111, projection='3d')
        axlim_min, axlim_max = X_pred_points[:, dim].min(), X_pred_points[:, dim].max()
        x, y = np.mgrid[axlim_min:axlim_max:complex(res.shape[0]),
                        1:0:complex(res.shape[1])]
        #x_dates = num2date(x)
        xfmt = mdates.DateFormatter('%b %d')
        ax.plot_surface(x,y,res,cmap=cmap,rstride=1, cstride=1, lw=0.05, alpha=1, edgecolor='b', norm=norm)
        #ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.set_zlabel('beta pdf')
        ax.set_ylabel('sentiment')
        ax.set_xlabel('date')
        #ax.autofmt_xdate()
        return fig, fig3d
