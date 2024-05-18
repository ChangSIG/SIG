import pandas as pd
from utils import NBS_core as core
import numpy as np


# filt the result from HN_90_cluster
def HN90_cluster_filter(nbs_org_ov_clinical_file, filt_clinical_file):
    clinical_data = pd.read_csv(nbs_org_ov_clinical_file, delimiter=',', header=0, index_col=None)
    TCGA_ID = np.array(clinical_data['TCGA_ID'])
    HN90_cluster_result = np.array(clinical_data['HN90_K4_cluster'])
    filt_clinical_data = pd.read_csv(filt_clinical_file, delimiter=',', header=0, index_col=None)
    filt_TCGA_ID = np.array(filt_clinical_data['TCGA_ID'])
    HN90_cluster_result_filt = []
    for i in range(len(TCGA_ID)):
        if TCGA_ID[i] in filt_TCGA_ID:
            HN90_cluster_result_filt.append(str(TCGA_ID[i]) + ',' + str(HN90_cluster_result[i]))
    with open('HN90_cluster_result_filt.csv', 'w') as f:
        for i in HN90_cluster_result_filt:
            f.write(i + '\n')


def pat_features_filter(pat_features_path, label_path):
    pats_features = pd.read_csv(pat_features_path, delimiter=',', index_col=0).astype(
        float)
    labels = pd.read_csv(label_path, delimiter='\t', index_col=0).astype(int)
    prop_data_qnorm = core.qnorm(pats_features)
    pats_features_filt = prop_data_qnorm.ix[labels.T.keys()]
    pats_features_filt.to_csv('patients_features_filt.csv')


# choose patients with clinical information
def sm_data_filter(sm_file_path, label_path):
    sm_data = pd.read_csv(sm_file_path, delimiter=',', index_col=0).astype(int)
    labels = pd.read_csv(label_path, delimiter='\t', index_col=0).astype(int)

    sm_data_filt = sm_data.ix[labels.T.keys()]
    sm_data_filt.to_csv('OV_sm_mat.csv')


def Clinical_filter(clinical_file_path, label_path):
    clinical = pd.read_csv(clinical_file_path, delimiter=',', index_col=0).astype(int)
    mut = pd.read_csv(label_path, delimiter=',', index_col=0)
    clinical_filt = clinical.ix[mut.T.keys()]

    clinical_filt.to_csv('clinical_filt.csv')


# Hofree_nbs clinicial data filter,to select 325 patients' samples from 328 samples
def Hofree_nbs_filter(cluster_result_file_path, clinical_file_path, K4M90_cluster_result_file_path):
    clinical = pd.read_csv(clinical_file_path, delimiter=',', index_col=0)
    cluster_result = pd.read_csv(cluster_result_file_path, delimiter=',', header=None)
    K4M90_cluster_result = pd.read_csv(cluster_result_file_path, delimiter=',', header=None, index_col=0)
    clinical_filt = clinical.ix[cluster_result.T.keys()]  # python2.7
    K4M90_cluster_result_filt = K4M90_cluster_result.ix[cluster_result.T.keys()]
    K4M90_cluster_result_filt.columns = ['cluster_assign']
    K4M90_cluster_filt = K4M90_cluster_result_filt['cluster_assign']

    K4M90_cluster_filt = np.array(K4M90_cluster_filt)
    with open('K4M90_filt.csv', 'w') as f:
        for i in K4M90_cluster_filt:
            f.write(str(i) + '\n')
    clinical_filt.to_csv('clinical_filt.csv')


def org_ov_clinical_filter(clinical_file):
    clinical = pd.read_csv(clinical_file, delimiter=',', index_col=0, header=0)
    clinical_filt = pd.DataFrame(clinical, columns=['vital_status', 'overall_survival'])
    clinical_filt.to_csv('clinical_data_filt.csv', sep=',')


def org_lung_clinical_filter(clinical_file):
    clinical = pd.read_csv(clinical_file, delimiter=',', index_col=0, header=0)
    # clinical_filt = pd.DataFrame(clinical, columns=['TCGA_ID', 'Vital_Status', 'Days_survival'])
    clinical_filt = pd.DataFrame(clinical, columns=['vital_status', 'overall_survival'])
    clinical_filt.to_csv('lung_clinical_data_filt.csv', sep=',')


# save cluster result and suivial information in same file
def clu_result_survial_data_conbine(cluster_file, survival_file, save_path):
    clu_result = pd.read_csv(cluster_file, delimiter=',', index_col=0, header=None)
    clu_result.columns = ['subtype']
    sur_data = pd.read_csv(survival_file, delimiter=',', index_col=0, header=0)
    conbine_file = sur_data.merge(clu_result, how='inner', left_index=True, right_index=True)
    conbine_file.to_csv(save_path)


if __name__ == '__main__':
    cluster_assign = 'OV_HN90_cluster_result.csv'
    clinical_survival = 'clinical_processed.csv'
    save_path = 'OV_HN90_conbine_file.csv'
    clu_result_survial_data_conbine(cluster_file=cluster_assign, survival_file=clinical_survival, save_path=save_path)