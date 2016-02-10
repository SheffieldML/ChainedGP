#include <math.h>

void _quad1d(int N, double* mu, double* var, double* Y, int Ngh, double* gh_x, double* gh_w, double* F, double* dF_dm, double* dF_dv){
    double x, std, logpdf, dlogpdf_df, d2logpdf_df2;
    int n,d;
    const double sqrt_2 = sqrt(2.0);
    const double normal_const = 1.0/(sqrt(2.0*M_PI));
    //#pragma omp parallel for private(d, x, logpdf, dlogpdf_df, d2logpdf_df2)
    for(n=0;n<N;n++){
        std = sqrt(2.0*var[n]);
        for(d=0; d<Ngh; d++){
            x = mu[n] + std*gh_x[d];
            logpdf = (1.0+erf(x/sqrt_2))/2.0; // Insert logpdf here
            dlogpdf_df = normal_const * exp(-(x*x)/2.0); //Insert dlogpdf_df here
            d2logpdf_df2 = logpdf/dlogpdf_df; //Insert d2logpdf_df2 here

            F[n] += gh_w[d]*logpdf;
            dF_dm[n] += gh_w[d]*dlogpdf_df;
            dF_dv[n] += gh_w[d]*d2logpdf_df2;
        }
    }
}

void _quad2d_stut(int N, double* muf, double* varf, double* mug, double* varg, double* Y, int Ngh, double v, double* gh_x, double* gh_w, double* F, double* dF_dmf, double* dF_dvf, double* dF_dmg, double* dF_dvg, double* dF_ddf){
    double Yi, gi, fj, stdfi, stdgi, mugi, mufi, logpdf, dlogpdf_df, d2logpdf_df2, dlogpdf_dg, d2logpdf_dg2, e, e2, e_gi, v_egi, dF_dquad_ddf;
    int n,j,i;
    //Asume a single output
    //#pragma omp parallel for private(n, fj, gi, i, j, logpdf, dlogpdf_df, d2logpdf_df2, dlogpdf_dg, d2logpdf_dg2)
    for(n=0;n<N;n++){
        //Get the points out that we will use, gi, fi, which will be shifted
        //and weighted
        stdfi = sqrt(2.0*varf[n]);
        stdgi = sqrt(2.0*varg[n]);
        mugi = mug[n];
        mufi = muf[n];
        Yi = Y[n];
        for(i=0; i<Ngh; i++){
            gi = mugi + stdgi*gh_x[i]; // Get the location in g scale
            e_gi = exp(gi);
            for(j=0; j<Ngh; j++){
                fj = mufi + stdfi*gh_x[j]; //Get the location in f scale
                e = Yi-fj; //Evaluate at f and g location
                e2 = pow(e, 2.0);
                v_egi = v*e_gi;
                logpdf = -0.5*(v+1)*log1p(e2/v_egi); // Insert logpdf here

                dlogpdf_df = (v+1)*(e) / (v_egi + e2); //Insert dlogpdf_df here
                d2logpdf_df2 = (v+1)*(e2 - v_egi) / pow(v_egi + e2, 2.0); //Insert d2logpdf_df2 here

                dlogpdf_dg = 0.5*(v+1)*( e2 / (v_egi + e2) ); //Insert dlogpdf_dg here
                d2logpdf_dg2 = -0.5*(v+1)*e2*v_egi / pow(v_egi + e2, 2.0); //Insert d2logpdf_dg2 here
                dF_dquad_ddf = 0.5*((v+1)*e2/(v*(v_egi + e2)) 
                                    - log1p(e2/v_egi));
                //Should be able to just pull in the gh_w[i] weights
                F[n] += gh_w[i]*gh_w[j]*logpdf;
                dF_dmf[n] += gh_w[i]*gh_w[j]*dlogpdf_df;
                dF_dvf[n] += gh_w[i]*gh_w[j]*d2logpdf_df2;
                dF_dmg[n] += gh_w[i]*gh_w[j]*dlogpdf_dg;
                dF_dvg[n] += gh_w[i]*gh_w[j]*d2logpdf_dg2;
                dF_ddf[n] += gh_w[i]*gh_w[j]*dF_dquad_ddf;
            }
        }
    }
}
