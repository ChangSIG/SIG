import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test as multiv_lr_test
import math


def cluster_color_assign(cluster_assignments, name=None):
    k = max(cluster_assignments.value_counts().index)
    colors = sns.color_palette('hls', k)
    cluster_cmap = {i: colors[i - 1] for i in range(1, k + 1)}
    pat_colors = {}
    for pat in cluster_assignments.index:
        pat_colors[pat] = cluster_cmap[cluster_assignments.loc[pat]]
    cluster_cmap = pd.Series(pat_colors, name=name)
    return cluster_cmap


def plot_cc_map(cc_table, linkage, row_color_map=None, col_color_map=None, verbose=True, **save_args):
    title = 'Co-Clustering Map'
    if 'job_name' in save_args:
        title = save_args['job_name'] + ' Co-Clustering Map'
    cg = sns.clustermap(cc_table, row_linkage=linkage, col_linkage=linkage,
                        cmap='Blues', cbar_kws={'label': 'Co-Cluster Frequency'},
                        row_colors=row_color_map, col_colors=col_color_map,
                        **{'xticklabels': 'False', 'yticklabels': 'False'})
    cg.cax.set_position([0.92, .11, .03, .584])
    cg.ax_heatmap.set_xlabel('')
    cg.ax_heatmap.set_xticks([])
    cg.ax_heatmap.set_ylabel('')
    cg.ax_heatmap.set_yticks([])
    cg.ax_row_dendrogram.set_visible(False)
    plt.suptitle(title, fontsize=20, x=0.6, y=0.95)
    if 'outdir' in save_args:
        if 'job_name' in save_args:
            save_cc_map_path = save_args['outdir'] + str(save_args['job_name']) + '_cc_map.png'
        else:
            save_cc_map_path = save_args['outdir'] + 'cc_map.png'
        plt.savefig(save_cc_map_path, bbox_inches='tight')
    if verbose:
        print('Co-Clustering Map plotted')
    return


def cluster_KMplot(cluster_assign, clin_data_fn, cancer_type, delimiter='\t', lr_test=True, tmax=-1, verbose=True, **save_args):
    title = 'KM Survival Plot'
    if 'job_name' in save_args:
        title = save_args['job_name'] + ' KM Survival Plot'

    kmf = KaplanMeierFitter()
    surv = pd.read_csv(clin_data_fn, sep=delimiter, index_col=0)
    clusters = sorted(list(cluster_assign.value_counts().index))
    k = len(clusters)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.subplot(1, 1, 1)
    colors = sns.color_palette('hls', k)
    cluster_cmap = {clusters[i]: colors[i] for i in range(k)}

    SCM_lst = []
    n_j_lst = []
    N = 0
    lifetime_way = 'rate'
    if cancer_type == 'OV':
        max_survival_days = 5480.0

    for clust in clusters:
        clust_pats = list(cluster_assign[cluster_assign == clust].index)
        clust_surv_data = pd.DataFrame(
            columns=['vital_status', 'days_to_death', 'days_to_last_followup', 'overall_survival'])
        for key in clust_pats:
            if key in surv.axes[0]:
                clust_surv_data = clust_surv_data.append(surv.loc[key]).fillna(0)
        fit_result = kmf.fit(clust_surv_data.overall_survival, clust_surv_data.vital_status, max_survival_days, label='Group ' + str(int(clust)))
        fit_result.alpha = 0.95
        kmf.plot(ax=ax, color=cluster_cmap[clust], ci_show=False)

        if lifetime_way == 'rate':
            k1 = (1.0 - 0.0) / (0.0 - 1.0)
            matrix1 = np.stack([fit_result.survival_function_.index[:-1],
                               fit_result.survival_function_.iloc[:-1, -1]])
            matrix2 = np.stack([fit_result.survival_function_.index[1:-1],
                                fit_result.survival_function_.iloc[2:, -1]])
            matrix = np.hstack((matrix1, matrix2))

        elif lifetime_way == 'month':
            k1 = (1.0 - 0.0) / (0.0 - (max_survival_days / 30.0))
            matrix = np.stack([fit_result.survival_function_.index / 30.0,
                               fit_result.survival_function_.iloc[:, -1]])
        else:
            k1 = (1.0 - 0.0) / (0.0 - max_survival_days)
            matrix = np.stack(
            [fit_result.survival_function_.index, fit_result.survival_function_.iloc[:, -1]])
        n_j = matrix.shape[-1]

        k2_sub = [matrix[:, 0] - matrix[:, i] for i in range(n_j)]
        k2_sub = np.array(k2_sub)[1:]
        k2 = np.array([(k2_sub[i, -1] / k2_sub[i, 0]) for i in range(k2_sub.shape[0])])
        SCM = np.sum(
            [np.arctan(np.abs((k2[i] - k1) / (1 + k1 * k2[i]))) * 180 / np.pi for i in range(k2_sub.shape[0])]) / (
                      n_j - 1)
        N += (n_j - 1)
        n_j_lst.append(n_j - 1)
        SCM_lst.append(SCM)
    SCM_avg = np.sum([(SCM_lst[i] * n_j_lst[i] / N) for i in range(len(SCM_lst))])
    print("lifetime_way:", lifetime_way)
    print("SCM_list: ", SCM_lst)
    print("SCM avg: ", SCM_avg)

    plt.xlim((0, 1.0))
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Survival Probability', fontsize=16)
    if lr_test:
        cluster_survivals = pd.concat([surv, cluster_assign], axis=1).dropna().astype(int)
        p = multiv_lr_test(np.array(cluster_survivals.overall_survival),
                           np.array(cluster_survivals[cluster_assign.name]), t_0=tmax,
                           event_observed=np.array(cluster_survivals.vital_status)).p_value
        if verbose:
            print('Multi-Class Log-Rank P:', p)
        plt.title(title + '\np=' + repr(round(p, 5)) + '   SCM avg=' + repr(round(SCM_avg, 4)), fontsize=20, y=1.02)
    else:
        plt.title(title, fontsize=24, y=1.02)
    if 'outdir' in save_args:
        if 'job_name' in save_args:
            save_KMplot_path = save_args['outdir'] + '/' + str(save_args['job_name']) + '_KM_plot.png'
        else:
            save_KMplot_path = save_args['outdir'] + '/' + 'KM_plot.png'
        plt.savefig(save_KMplot_path, bbox_inches='tight')
    if verbose:
        print('Kaplan Meier Plot constructed')
    if lr_test:
        return p, SCM_avg
    else:
        return p, SCM_avg
        
        
def cluster_KMplot_cindex(cluster_assign, clin_data_fn, cancer_type, delimiter='\t', lr_test=True, tmax=-1, verbose=True, **save_args):
    title = 'KM Survival Plot'
    if 'job_name' in save_args:
        title = save_args['job_name'] + ' KM Survival Plot'

    kmf = KaplanMeierFitter()
    cph = CoxPHFitter()
    surv = pd.read_csv(clin_data_fn, sep=delimiter, index_col=0)
    clusters = sorted(list(cluster_assign.value_counts().index))
    k = len(clusters)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.subplot(1, 1, 1)
    colors = sns.color_palette('hls', k)
    cluster_cmap = {clusters[i]: colors[i] for i in range(k)}

    combined_df = pd.concat([surv, cluster_assign], axis=1).dropna().astype(int)
    combined_df = combined_df.reset_index()
    combined_df = combined_df.rename(columns={'index': 'TCGA_ID', combined_df.columns[-1]: 'cluster_result'})
    combined_df = combined_df.loc[:, ['TCGA_ID', 'vital_status', 'overall_survival', 'cluster_result']]
    combined_df['cluster_result'] = combined_df['cluster_result'].astype('category')
    cph.fit(combined_df, duration_col='overall_survival', event_col='vital_status', formula="cluster_result")
    
    plt.xlim((0, 1.0))
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Survival Probability', fontsize=16)
    if lr_test:
        cluster_survivals = pd.concat([surv, cluster_assign], axis=1).dropna().astype(int)
        
        p = multiv_lr_test(np.array(cluster_survivals.overall_survival),
                           np.array(cluster_survivals[cluster_assign.name]), t_0=tmax,
                           event_observed=np.array(cluster_survivals.vital_status)).p_value
        if verbose:
            print('Multi-Class Log-Rank P:', p)
        plt.title(title + '\np=' + repr(round(p, 5)), fontsize=20, y=1.02)
    else:
        plt.title(title, fontsize=24, y=1.02)
    if 'outdir' in save_args:
        if 'job_name' in save_args:
            save_KMplot_path = save_args['outdir'] + '/' + str(save_args['job_name']) + '_KM_plot.png'
        else:
            save_KMplot_path = save_args['outdir'] + '/' + 'KM_plot.png'
        plt.savefig(save_KMplot_path, bbox_inches='tight')
    if verbose:
        print('Kaplan Meier Plot constructed')
    if lr_test:
        return p, cph.concordance_index_
    else:
        return p, cph.concordance_index_
