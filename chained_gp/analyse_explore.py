#Script to unravel the tasks and save them out to dataframes
import jug
import numpy as np
from functools import partial
import datetime
import subprocess

git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])[:6]

jug_dict = jug.init('experiments_explore.py', 'experiments_explore.jugdata/')[1]
exs_stut = jug_dict['stut_experiments']
exs_gauss = jug_dict['gauss_experiments']
exs_survival = jug_dict['survival_experiments']

exs = exs_stut + exs_gauss + exs_survival
filename = 'results/experiments_explore_data_all-{}-{}.pkl'.format(datetime.datetime.now().strftime('%d%m%y_%H:%M:%S'), git_hash)

es = []
i = 0
for e in exs:
    try:
        es.append(jug.value(e.__dict__))
    except Exception, e:
        print "Something went wrong"
        print e
        print i
        i += 1

#es are all our experiments wrapped up
#Lets make a dataframe of results
import pandas as pd
exs_df = pd.DataFrame(columns=['fold', 'seed', 'num_inducing', 'fixZ', 'f_bias', 'g_bias', 'f_rbf_len', 'g_rbf_len', 'g_mean', 'f_rbf_var'])
for e in es:
    default_d = {'fold':e['fold'],
                 'seed':e['seed'],
                 'num_inducing':e['num_inducing'],
                 'fixZ':e['fixZ'],
                 'f_bias':e['f_bias'],
                 'g_bias':e['g_bias'],
                 'f_rbf_len':e['f_rbf_len'],
                 'g_rbf_len':e['g_rbf_len'],
                 'f_rbf_var':e['f_rbf_var'],
                 'g_mean':e['g_mean'],
                 'dataset':e['dataset'],
		 'dataset_name':e['d_name'],
		 'optimize_df':e['optimize_df'],
                 'Z':e['Z']}

    #Make a row for each data
    gauss_d = default_d.copy()
    stut_d = default_d.copy()
    laplace_stut_d = default_d.copy()
    survival_d = default_d.copy()
    laplace_survival_d = default_d.copy()
    multi_gauss_d = default_d.copy()
    multi_stut_d = default_d.copy()
    multi_survival_d = default_d.copy()

    gauss_d['model_type'] = 'gauss'
    gauss_d['opt_params'] = e['m_gauss_opt_params']
    gauss_d['log_pred'] = e['m_gauss_log_pred']
    gauss_d['RMSE'] = e['m_gauss_RMSE']
    gauss_d['MAE'] = e['m_gauss_MAE']
    gauss_d['log_likelihood'] = e['m_gauss_log_likelihood']
    gauss_d['prediction'] = e['m_gauss_predictions']

    multi_gauss_d['model_type'] = 'multi_gauss'
    multi_gauss_d['opt_params'] = e['m_multi_gauss_opt_params']
    multi_gauss_d['log_pred'] = e['m_multi_gauss_log_pred']
    multi_gauss_d['RMSE'] = e['m_multi_gauss_RMSE']
    multi_gauss_d['MAE'] = e['m_multi_gauss_MAE']
    multi_gauss_d['log_likelihood'] = e['m_multi_gauss_log_likelihood']
    multi_gauss_d['prediction'] = e['m_multi_gauss_predictions']

    stut_d['model_type'] = 'stut'
    stut_d['opt_params'] = e['m_stut_opt_params']
    stut_d['log_pred'] = e['m_stut_log_pred']
    stut_d['RMSE'] = e['m_stut_RMSE']
    stut_d['MAE'] = e['m_stut_MAE']
    stut_d['log_likelihood'] = e['m_stut_log_likelihood']
    stut_d['prediction'] = e['m_stut_predictions']

    laplace_stut_d['model_type'] = 'laplace_stut'
    laplace_stut_d['opt_params'] = e['m_laplace_stut_opt_params']
    laplace_stut_d['log_pred'] = e['m_laplace_stut_log_pred']
    laplace_stut_d['RMSE'] = e['m_laplace_stut_RMSE']
    laplace_stut_d['MAE'] = e['m_laplace_stut_MAE']
    laplace_stut_d['log_likelihood'] = e['m_laplace_stut_log_likelihood']
    laplace_stut_d['prediction'] = e['m_laplace_stut_predictions']

    multi_stut_d['model_type'] = 'multi_stut'
    multi_stut_d['opt_params'] = e['m_multi_stut_opt_params']
    multi_stut_d['log_pred'] = e['m_multi_stut_log_pred']
    multi_stut_d['RMSE'] = e['m_multi_stut_RMSE']
    multi_stut_d['MAE'] = e['m_multi_stut_MAE']
    multi_stut_d['log_likelihood'] = e['m_multi_stut_log_likelihood']
    multi_stut_d['prediction'] = e['m_multi_stut_predictions']

    survival_d['model_type'] = 'survival'
    survival_d['opt_params'] = e['m_survival_opt_params']
    survival_d['log_pred'] = e['m_survival_log_pred']
    survival_d['RMSE'] = e['m_survival_RMSE']
    survival_d['MAE'] = e['m_survival_MAE']
    survival_d['log_likelihood'] = e['m_survival_log_likelihood']
    survival_d['prediction'] = e['m_survival_predictions']

    laplace_survival_d['model_type'] = 'laplace_survival'
    laplace_survival_d['opt_params'] = e['m_laplace_survival_opt_params']
    laplace_survival_d['log_pred'] = e['m_laplace_survival_log_pred']
    laplace_survival_d['RMSE'] = e['m_laplace_survival_RMSE']
    laplace_survival_d['MAE'] = e['m_laplace_survival_MAE']
    laplace_survival_d['log_likelihood'] = e['m_laplace_survival_log_likelihood']
    laplace_survival_d['prediction'] = e['m_laplace_survival_predictions']

    multi_survival_d['model_type'] = 'multi_survival'
    multi_survival_d['opt_params'] = e['m_multi_survival_opt_params']
    multi_survival_d['log_pred'] = e['m_multi_survival_log_pred']
    multi_survival_d['RMSE'] = e['m_multi_survival_RMSE']
    multi_survival_d['MAE'] = e['m_multi_survival_MAE']
    multi_survival_d['log_likelihood'] = e['m_multi_survival_log_likelihood']
    multi_survival_d['prediction'] = e['m_multi_survival_predictions']

    exs_df = exs_df.append(gauss_d, ignore_index=True)
    exs_df = exs_df.append(stut_d, ignore_index=True)
    exs_df = exs_df.append(survival_d, ignore_index=True)
    exs_df = exs_df.append(laplace_stut_d, ignore_index=True)
    exs_df = exs_df.append(laplace_survival_d, ignore_index=True)
    exs_df = exs_df.append(multi_gauss_d, ignore_index=True)
    exs_df = exs_df.append(multi_stut_d, ignore_index=True)
    exs_df = exs_df.append(multi_survival_d, ignore_index=True)

pd.save(exs_df, filename )

def print_results(df):
    means_multi_stut_log_pred = df['m_multi_stut_log_pred'].apply(np.mean)
    means_stut_log_pred = df['m_stut_log_pred'].apply(np.mean)
    means_gauss_log_pred = df['m_gauss_log_pred'].apply(np.mean)
    means_multi_gauss_log_pred = df['m_multi_gauss_log_pred'].apply(np.mean)

    means_multi_stut_RMSE = df['m_multi_stut_RMSE'].apply(np.mean)
    means_stut_RMSE = df['m_stut_RMSE'].apply(np.mean)
    means_gauss_RMSE = df['m_gauss_RMSE'].apply(np.mean)
    means_multi_gauss_RMSE = df['m_multi_gauss_RMSE'].apply(np.mean)

    means_multi_stut_MAE = df['m_multi_stut_MAE'].apply(np.mean)
    means_stut_MAE = df['m_stut_MAE'].apply(np.mean)
    means_gauss_MAE = df['m_gauss_MAE'].apply(np.mean)
    means_multi_gauss_MAE = df['m_multi_gauss_MAE'].apply(np.mean)

    print "Max multilatent student t log pred: {}".format(means_multi_stut_log_pred.max())
    print "Max student t log pred: {}".format(means_stut_log_pred.max())
    print "Max gauss log pred: {}".format(means_gauss_log_pred.max())
    print "Max multi log pred: {}".format(means_multi_gauss_log_pred.max())
    print ""

    print "min multilatent student t RMSE: {}".format(means_multi_stut_RMSE.min())
    print "min student t RMSE: {}".format(means_stut_RMSE.min())
    print "min gauss RMSE: {}".format(means_gauss_RMSE.min())
    print "min multi RMSE: {}".format(means_multi_gauss_RMSE.min())
    print ""

    print "min multilatent student t MAE: {}".format(means_multi_stut_MAE.min())
    print "min student t MAE: {}".format(means_stut_MAE.min())
    print "min gauss MAE: {}".format(means_gauss_MAE.min())
    print "min multi MAE: {}".format(means_multi_gauss_MAE.min())

