import numpy as np
cimport numpy as np
#from scipy.special import gammaln

ctypedef np.float64_t DTYPE_t

cdef extern from "quad_utils.h":
    void _quad1d "_quad1d" (int N, double* mu, double* var, double* Y, int Ngh, double* gh_x, double* gh_w, double* F, double* dF_dm, double* dF_dv)
cdef extern from "quad_utils.h":
    void _quad2d_stut "_quad2d_stut" (int N, double* muf, double* varf, double* mug, double* varg, double* Y, int Ngh, double v, double* gh_x, double* gh_w, double* F, double* dF_dmf, double* dF_dvf, double* dF_dmg, double* dF_dvg, double* dF_ddf)

cdef extern from "math.h":
    double exp(double x)
cdef extern from "math.h":
    double sqrt(double x)

def quad2d_stut(int N, np.ndarray[DTYPE_t, ndim=1] _muf,
                np.ndarray[DTYPE_t, ndim=1] _varf,
                np.ndarray[DTYPE_t, ndim=1] _mug,
                np.ndarray[DTYPE_t, ndim=1] _varg,
                np.ndarray[DTYPE_t, ndim=1] _Y,
                int Ngh,
                double df,
                np.ndarray[DTYPE_t, ndim=1] _gh_x,
                np.ndarray[DTYPE_t, ndim=1] _gh_w,
                np.ndarray[DTYPE_t, ndim=1] _F,
                np.ndarray[DTYPE_t, ndim=1] _dF_dmf,
                np.ndarray[DTYPE_t, ndim=1] _dF_dvf,
                np.ndarray[DTYPE_t, ndim=1] _dF_dmg,
                np.ndarray[DTYPE_t, ndim=1] _dF_dvg,
                np.ndarray[DTYPE_t, ndim=1] _dF_ddf
        ):
    cdef double *muf = <double*> _muf.data
    cdef double *varf = <double*> _varf.data
    cdef double *mug = <double*> _mug.data
    cdef double *varg = <double*> _varg.data
    cdef double *Y = <double*> _Y.data
    cdef double *gh_x = <double*> _gh_x.data
    cdef double *gh_w = <double*> _gh_w.data
    cdef double *F = <double*> _F.data
    cdef double *dF_dmf = <double*> _dF_dmf.data
    cdef double *dF_dvf = <double*> _dF_dvf.data
    cdef double *dF_dmg = <double*> _dF_dmg.data
    cdef double *dF_dvg = <double*> _dF_dvg.data
    cdef double *dF_ddf = <double*> _dF_ddf.data
    _quad2d_stut(N, muf, varf, mug, varg, Y, Ngh, df, gh_x, gh_w, F, dF_dmf, dF_dvf, dF_dmg, dF_dvg, dF_ddf)

def quad2d_beta(double[:, :] muf,
                double[:, :] varf,
                double[:, :] mug,
                double[:, :] varg,
                double[:, :] Y,
                double[:] gh_x,
                double[:] gh_w,
                double[:, :] F,
                double[:, :] dF_dmf,
                double[:, :] dF_dvf,
                double[:, :] dF_dmg,
                double[:, :] dF_dvg,
        ):
    cdef int N = Y.shape[0]
    cdef int D = Y.shape[1]
    cdef int Ngh = gh_x.shape[0]
    cdef int n, i, j
    for d in range(D):
        for n in range(N):
            stdfi = sqrt(2.0*varf[n, d])
            stdgi = sqrt(2.0*varg[n, d])
            mugi = mug[n, d]
            mufi = muf[n, d]
            Yi = Y[n, d]
            for i in range(Ngh):
                #Use c sqrt instead of numpy
                gi = mugi + stdgi*gh_x[i]
                e_gi = exp(gi)
                for j in range(Ngh):
                    fj = mufi + stdfi*gh_x[j] #Get the location in f scale
                    e_fj = exp(fj)
                    #logpdf = -lgam(e_fj) -lgam(e_gi) + lgam(e_fj + e_gi)
                    raise NotImplementedError
                    logpdf = Gamma(Yi)
                    dlogpdf_df = polygamma(0, Yi)
                    d2logpdf_df2 = 0.0
                    dlogpdf_dg = 0.0
                    d2logpdf_dg2 = 0.0

                    F[n, d] += gh_w[i]*gh_w[j]*logpdf
                    dF_dmf[n, d] += gh_w[i]*gh_w[j]*dlogpdf_df
                    dF_dvf[n, d] += gh_w[i]*gh_w[j]*d2logpdf_df2
                    dF_dmg[n, d] += gh_w[i]*gh_w[j]*dlogpdf_dg
                    dF_dvg[n, d] += gh_w[i]*gh_w[j]*d2logpdf_dg2
    return F, dF_dmf, dF_dvf, dF_dmg, dF_dvg
