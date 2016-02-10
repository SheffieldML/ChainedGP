from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.util import linalg
from GPy.util import choleskies
import numpy as np
from GPy.inference.latent_function_inference.posterior import Posterior
from scipy.linalg.blas import dgemm, dsymm, dtrmm

from collections import namedtuple
#LatentFunctionDetails = namedtuple("LatentDetails", "q_u_mean q_u_chol mean_function mu v prior_mean_u L A S Kmm Kmmi Kmmim Kmmi_mfZ KL dKL_dm dKL_dS dKL_dKmm dKL_dmfZ")
LatentFunctionDetails = namedtuple("LatentDetails", "q_u_mean q_u_chol mean_function mu v prior_mean_u L A S Kmm Kmmi Kmmim KL")

class SVGPMultiInf(LatentFunctionInference):

    def inference(self, q_u_means, q_u_chols, kern, X, Z, likelihood, Y, mean_functions, Y_metadata=None, KL_scale=1.0, batch_scale=1.0):
        num_inducing = Z.shape[0]
        num_data, num_outputs = Y.shape
        num_latent_funcs = likelihood.request_num_latent_functions(Y)

        #For each latent function, calculate some required values
        latent_function_details = []
        for latent_ind in range(num_latent_funcs):
            q_u_meanj = q_u_means[:, latent_ind*num_outputs:(latent_ind+1)*num_outputs]
            q_u_cholj = q_u_chols[:, latent_ind*num_outputs:(latent_ind+1)*num_outputs]
            kernj = kern[latent_ind]
            mean_functionj = mean_functions[latent_ind]
            latent_detail = self.calculate_mu_var(X, Y, Z, q_u_meanj, q_u_cholj, kernj, mean_functionj, num_inducing, num_data, num_outputs)
            latent_function_details.append(latent_detail)

        mu = np.hstack([l.mu for l in latent_function_details])
        v = np.hstack([l.v for l in latent_function_details])
        #mu = [l.mu for l in latent_function_details]
        #v = [l.v for l in latent_function_details]

        #Hack shouldn't be necessary
        #Y = np.hstack([Y]*num_latent_funcs)

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)

        #for latent_ind in range(num_latent_functions):
            #l.dF_dmu = dF_dmu[:, latent_ind][:, None]
            #l.dF_dv = dF_dv[:, latent_ind][:, None]

        #rescale the F term if working on a batch
        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
        if dF_dthetaL is not None:
            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale

        #sum (gradients of) expected likelihood and KL part
        log_marginal = F.sum()

        dL_dKmm = []
        dL_dKmn = []
        dL_dKdiag = []
        dL_dm = []
        dL_dchol = []
        dL_dmfZ = []
        dL_dmfX = []
        posteriors = []
        #For each latent function (and thus for each kernel the latent function uses)
        #calculate the gradients and generate a posterior
        for latent_ind in range(num_latent_funcs):
            l = latent_function_details[latent_ind]
            #q_u_meanj = q_u_means[:, latent_ind*num_outputs:(latent_ind+1)*num_outputs]
            #q_u_cholj = q_u_chols[:, latent_ind*num_outputs:(latent_ind+1)*num_outputs]
            dF_dmui = dF_dmu[:, latent_ind*num_outputs:(latent_ind+1)*num_outputs]
            dF_dvi = dF_dv[:, latent_ind*num_outputs:(latent_ind+1)*num_outputs]
            (log_marginal, dL_dKmmi, dL_dKmni, dL_dKdiagi,
            dL_dmi, dL_dcholi, dL_dmfZi, dL_dmfXi) = self.calculate_gradients(log_marginal, l, dF_dmui, dF_dvi,
                                                                              num_inducing, num_outputs, num_data)
            posterior = Posterior(mean=l.q_u_mean, cov=l.S.T, K=l.Kmm, prior_mean=l.prior_mean_u)

            dL_dKmm.append(dL_dKmmi)
            dL_dKmn.append(dL_dKmni)
            dL_dKdiag.append(dL_dKdiagi)
            dL_dm.append(dL_dmi)
            dL_dchol.append(dL_dcholi)
            dL_dmfZ.append(dL_dmfZi)
            dL_dmfX.append(dL_dmfXi)
            posteriors.append(posterior)

        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dL_dKdiag, 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}

        #If not all of the mean functions are null, fill out the others gradients with zeros
        if not all(mean_function is None for mean_function in mean_functions):
            for mean_function in mean_functions:
                if mean_function is None:
                    grad_dict['dL_dmfZ'] = np.zeros(Z.shape)
                    grad_dict['dL_dmfX'] = np.zeros(X.shape)
                else:
                    grad_dict['dL_dmfZ'] = dL_dmfZ
                    grad_dict['dL_dmfX'] = dL_dmfX

        return posteriors, log_marginal, grad_dict

    def calculate_gradients(self, log_marginal, latent_info, dF_dmu, dF_dv, num_inducing, num_outputs, num_data):
        """
        Given a named tuple for lots of parameters of the latent function, calculate the
        gradients wrt to its latent functions and kernel
        """
        l = latent_info
        #derivatives of expected likelihood, assuming zero mean function
        #Adv = l.A.T[:, :, None]*dF_dv[None, :, :] # As if dF_Dv is diagonal
        Adv = l.A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
        #Admu = l.A.T.dot(dF_dmu)
        Admu = l.A.dot(dF_dmu)
        #AdvA = np.dstack([np.dot(l.A.T, Adv[:,:,i].T) for i in range(num_outputs)])
        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
        AdvA = np.dot(Adv.reshape(-1, num_data),l.A.T).reshape(num_outputs, num_inducing, num_inducing )
        #tmp = linalg.ijk_jlk_to_il(AdvA, l.S).dot(l.Kmmi)
        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, l.S)],0).dot(l.Kmmi)
        #dF_dKmm = -Admu.dot(l.Kmmim.T) + AdvA.sum(-1) - tmp - tmp.T
        dF_dKmm = -Admu.dot(l.Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        #tmp = 2.*(linalg.ij_jlk_to_ilk(l.Kmmi, l.S) - np.eye(num_inducing)[:,:,None])
        tmp = l.S.reshape(-1, num_inducing).dot(l.Kmmi).reshape(num_outputs, num_inducing , num_inducing )
        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])
        #dF_dKmn = linalg.ijk_jlk_to_il(tmp, Adv) + l.Kmmim.dot(dF_dmu.T)
        dF_dKmn = l.Kmmim.dot(dF_dmu.T)
        for a,b in zip(tmp, Adv):
            dF_dKmn += np.dot(a.T, b)
        dF_dm = Admu
        dF_dS = AdvA

        #gradient of the KL term (assuming zero mean function)
        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        Si = choleskies.multiple_dpotri(l.L)

        if np.any(np.isinf(Si)):
            raise ValueError("Cholesky representation unstable")
            #S = S + np.eye(S.shape[0])*1e-5*np.max(np.max(S))
            #Si, Lnew, _,_ = linalg.pdinv(S)

        dKL_dm = l.Kmmim.copy()
        #dKL_dS = 0.5*(l.Kmmi[:,:,None] - Si)
        dKL_dS = 0.5*(l.Kmmi[None,:,:] - Si)
        #dKL_dKmm = 0.5*num_outputs*l.Kmmi - 0.5*l.Kmmi.dot(l.S.sum(-1)).dot(l.Kmmi) - 0.5*l.Kmmim.dot(l.Kmmim.T)
        dKL_dKmm = 0.5*num_outputs*l.Kmmi - 0.5*l.Kmmi.dot(l.S.sum(0)).dot(l.Kmmi) - 0.5*l.Kmmim.dot(l.Kmmim.T)

        #adjust gradient to account for mean function
        dL_dmfZ = None
        dL_dmfX = None
        KL = l.KL
        if l.mean_function is not None:
            #adjust KL term for mean function
            Kmmi_mfZ = np.dot(l.Kmmi, l.prior_mean_u)
            KL += -np.sum(l.q_u_mean*Kmmi_mfZ)
            KL += 0.5*np.sum(Kmmi_mfZ*l.prior_mean_u)

            #adjust gradient for mean fucntion
            dKL_dm -= Kmmi_mfZ
            dKL_dKmm += l.Kmmim.dot(Kmmi_mfZ.T)
            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)

            #compute gradients for mean_function
            dKL_dmfZ = Kmmi_mfZ - l.Kmmim

            dF_dmfX = dF_dmu.copy()
            dF_dmfZ = -Admu
            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
            dF_dKmm += Admu.dot(Kmmi_mfZ.T)

            dL_dmfZ = dF_dmfZ - dKL_dmfZ
            dL_dmfX = dF_dmfX

        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS - dKL_dS, dF_dKmm - dKL_dKmm, dF_dKmn

        #dL_dchol = np.dstack([2.*np.dot(dL_dS[:, :, i], l.L[: , :,i]) for i in range(num_outputs)])
        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, l.L) ])
        dL_dchol = choleskies.triang_to_flat(dL_dchol)

        log_marginal -= KL

        dL_dKdiag = dF_dv.sum(1)
        return log_marginal, dL_dKmm, dL_dKmn, dL_dKdiag, dL_dm, dL_dchol, dL_dmfZ, dL_dmfX

    def calculate_mu_var(self, X, Y, Z, q_u_mean, q_u_chol, kern, mean_function, num_inducing, num_data, num_outputs):
        """
        Calculate posterior mean and variance for the latent function values for use in the
        expectation over the likelihood
        """
        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol)
        #S = linalg.ijk_ljk_to_ilk(L, L) #L.dot(L.T)
        S = np.empty((num_outputs, num_inducing, num_inducing))
        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
        #logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[:,:,i])))) for i in range(L.shape[-1])])
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])
        #compute mean function stuff
        if mean_function is not None:
            prior_mean_u = mean_function.f(Z)
            prior_mean_f = mean_function.f(X)
        else:
            prior_mean_u = np.zeros((num_inducing, num_outputs))
            prior_mean_f = np.zeros((num_data, num_outputs))

        #compute kernel related stuff
        Kmm = kern.K(Z)
        #Knm = kern.K(X, Z)
        Kmn = kern.K(Z, X)
        Knn_diag = kern.Kdiag(X)
        #Kmmi, Lm, Lmi, logdetKmm = linalg.pdinv(Kmm)
        Lm = linalg.jitchol(Kmm)
        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
        Kmmi, _ = linalg.dpotri(Lm)

        #compute the marginal means and variances of q(f)
        #A = np.dot(Knm, Kmmi)
        A, _ = linalg.dpotrs(Lm, Kmn)
        #mu = prior_mean_f + np.dot(A, q_u_mean - prior_mean_u)
        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
        #v = Knn_diag[:,None] - np.sum(A*Knm,1)[:,None] + np.sum(A[:,:,None] * linalg.ij_jlk_to_ilk(A, S), 1)
        v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
            v[:,i] = np.sum(np.square(tmp),0)
        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]

        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        #KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.einsum('ij,ijk->k', Kmmi, S) + 0.5*np.sum(q_u_mean*Kmmim,0)
        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
        KL = KLs.sum()

        latent_detail = LatentFunctionDetails(q_u_mean=q_u_mean, q_u_chol=q_u_chol, mean_function=mean_function,
                                              mu=mu, v=v, prior_mean_u=prior_mean_u, L=L, A=A,
                                              S=S, Kmm=Kmm, Kmmi=Kmmi, Kmmim=Kmmim, KL=KL)
        return latent_detail
