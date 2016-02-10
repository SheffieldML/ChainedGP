//#include <omp.h>
#include <math.h>
void _quad1d(int N, double* mu, double* var, double* Y, int Ngh, double* gh_x, double* gh_w, double* F, double* dF_dm, double* dF_dv);
void _quad2d_stut(int N, double* muf, double* varf, double* mug, double* varg, double* Y, int Ngh, double v, double* gh_x, double* gh_w, double* F, double* dF_dmf, double* dF_dvf, double* dF_dmg, double* dF_dvg, double* dF_ddf);
