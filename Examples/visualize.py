from utils import plot

import os
import time
import pandas as pd
import numpy as np
from IPython.display import Image

OV_predict_cluster = pd.read_csv('Examples/Results/OV_cluster.csv', delimiter=',', header=None, index_col=0)

OV_predict_cluster.columns = ['cluster_assign']
OV_cluster_assign = OV_predict_cluster['cluster_assign']
OV_surv_data_path = 'Examples/Example_Data/Clinical_Files/ov_clinical.csv'

save_args = {'outdir': 'Examples/output/OV/', 'job_name': 'OV_keep1.0_merged'}
print('job_name: ' + save_args['job_name'] + ':')
# Plot KM Plot for patient clusters
my_ov_cluster = plot.cluster_KMplot(OV_cluster_assign, OV_surv_data_path, cancer_type='OV', delimiter=',', **save_args)
Image(filename=save_args['outdir'] + save_args['job_name'] + '_KM_plot.png', width=600, height=600)

# true_label_path = '../Examples/Example_Data/processed_data/K4M90_Patient_Label_raw.txt'
# true_label_path = '../Examples/Example_Data/processed_data/K4M90_filt_325.csv'
# true_label = np.loadtxt(true_label_path, dtype=int)
# ari = adjusted_rand_score(true_label, np.array(OV_cluster_assign))
# print('ari:{:.4f}'.format(ari))
