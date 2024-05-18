import sys, os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

from utils import core, prop, NBS_single, cc, plot, dit

import argparse
import os
import time
import pandas as pd
import numpy as np
import torch

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# import mkl


# Valid file path check (Does not check file formatting, but checks if given path exists and is readable)
def valid_infile(in_file):
    if not os.path.isfile(in_file):
        raise argparse.ArgumentTypeError("{0} is not a valid input file path".format(in_file))
    if os.access(in_file, os.R_OK):
        return in_file
    else:
        raise argparse.ArgumentTypeError("{0} is not a readable input file".format(in_file))


# Checking valid integer values (for all values that must be >0)
def positive_int(x):
    x = int(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("%s must be a positive integer" % x)
    return x


# Checking valid alpha and p values (Range is 0.0-1.0 exclusive)
def restricted_float(x):
    x = float(x)
    if (x <= 0.0) or (x >= 1.0):
        # raise arg
        print('invalid x')
    return x


def main():
    parser = argparse.ArgumentParser(
        description='Perform network based stratification of tumors by binary somatic mutation profile on given molecular network. Optional inputs will overrride any corresponding values set by configuration parameters file (if it is given)')

    # Required files from user
    required_inputs = parser.add_argument_group('Files required by NBS')
    required_inputs.add_argument("-sm_data_file", '--sm_data_file', type=valid_infile,
                                 default='Examples/Example_Data/Mutation_Files/OV_sm_mat_data.csv')
    parser.add_argument('-j', '--job_name', type=str, required=False, default='OV_keep1.0_merged')
    required_inputs.add_argument("-network_file", '--network_file', type=valid_infile,
                                 default='Examples/Example_Data/processed_data/graph/OV_keep1.0_merged_graph.txt')

    parser.add_argument('-surv', '--survival_data', type=valid_infile, required=False,
                        default='Examples/Example_Data/Clinical_Files/ov_clinical.csv')

    parser.add_argument('-k', '--K', type=positive_int, required=False, default=4,)
    parser.add_argument('-n', '--niter', type=positive_int, required=False, default=100)
    parser.add_argument('-o', '--outdir', type=str, required=False, default='Examples/output/')

    # Other optional run parameters
    parser.add_argument('-params', '--params_file', type=valid_infile, default=None, required=False,)
    parser.add_argument('-a', '--alpha', type=restricted_float, required=False, default=0.7,)
    parser.add_argument('-t', '--threads', type=positive_int, default=1, required=False,)
    parser.add_argument('-nv', '--no_verbose', default=False, action="store_true", required=False,)
    # parser.add_argument('-sp', '--save_prop', type=str, required=True,
    #                     help=' ')
    # Load NBS arguments from parser
    args = parser.parse_args()

    # # Set the number of threads to use
    # mkl.set_num_threads(args.threads)

    # Load NBS intermediate parameters (from file if applicable, otherwise use all default values)
    if args.params_file is None:
        params = dit.load_params()
    else:
        params = dit.load_params(params_file=args.params_file)

    # Note about command line arguments passed:If a configuration parameter file is passed and it does not match the corresponding parameter value given by the command line,
    # NBS will override the parameter value in the parameter file.
    # Overriding parameter values with command line inputs (if applicable)
    if args.outdir is not None:
        params['outdir'] = args.outdir
    if args.job_name is not None:
        params['job_name'] = args.job_name
    if args.alpha is not None:
        params['prop_alpha'] = args.alpha
    if args.K is not None:
        params['netNMF_k'] = args.K
    if args.niter is not None:
        params['niter'] = args.niter
    if args.survival_data is not None:
        params['plot_survival'] = True
    else:
        params['plot_survival'] = False
    if args.no_verbose:
        verbose = False
    else:
        verbose = params['verbose']

    params['mut_filetype'] = 'matrix'
    params['mut_filedelim'] = ','
    params['net_filedelim'] = '\t'
    params['surv_file_delim'] = ','
    # params['prop_symmetric_norm'] = True  # (default=false)
    params['pats_subsample_p'] = 0.8
    params['gene_subsample_p'] = 0.8
    params['min_muts'] = 2

    # Setting up NBS run

    # Constructing output directory if directory does not exist
    if not os.path.exists(params['outdir']):
        os.makedirs(params['outdir'])
    # Setting up save file parameters
    save_args = {'outdir': params['outdir'], 'job_name': params['job_name']}

    # Begin NBS
    print()
    print('##################################################################################')
    print('# Beginning NBS Run:', params['job_name'])
    print('##################################################################################')
    print('Results ouput directory:', save_args['outdir'])
    print()
    print('##################################################################################')
    print('# Loading binary somatic mutation data')
    print('##################################################################################')
    if verbose:
        print('Binary somatic mutation data file type:', params['mut_filetype'])
        print('Binary somatic mutation data file delimiter:', repr(params['mut_filedelim']))
    sm_mat = dit.load_binary_mutation_data(args.sm_data_file, filetype=params['mut_filetype'],
                                           delimiter=params['mut_filedelim'], verbose=verbose)
    print()
    print('##################################################################################')
    print('# Loading network from file')
    print('##################################################################################')
    if verbose:
        print('Network file delimiter:', repr(params['net_filedelim']))
        print('Degree-preserved shuffle of network:', params['degree_preserved_shuffle'])
        print('Node label shuffle of network:', params['node_label_shuffle'])
    network = dit.load_network_file(args.network_file, delimiter=params['net_filedelim'],
                                    degree_shuffle=params['degree_preserved_shuffle'],
                                    label_shuffle=params['node_label_shuffle'], verbose=verbose)
    print()
    print('##################################################################################')
    print('# Construct regularization network graph laplacian for network-regularized NMF')
    print('##################################################################################')
    if verbose:
        print('Input network laplacian diagonal offset (gamma):', params['reg_net_gamma'])
        print('Number of nearest neighbors to connect each network node to for regularization network:', params[
            'k_nearest_neighbors'])
        print('Save regularization network graph laplacian:', params['save_knn_glap'])
        if params['save_knn_glap']:
            print(save_args['outdir'] + str(save_args['job_name']) + '_knnGlap.csv')
    if params['save_knn_glap']:
        knnGlap = core.network_inf_KNN_glap(network, gamma=params['reg_net_gamma'], kn=params['k_nearest_neighbors'],
                                            verbose=verbose, **save_args)
    else:
        knnGlap = core.network_inf_KNN_glap(network, gamma=params['reg_net_gamma'], kn=params['k_nearest_neighbors'],
                                            verbose=verbose)
    print()
    print('##################################################################################')
    print('# Construct network propagation kernel')
    print('##################################################################################')
    if verbose:
        print('Network propagation coefficient (alpha):', params['prop_alpha'])
        print('Symmetric adjacency matrix normalization for propagation:', params['prop_symmetric_norm'])
        print('Save network propagation kernel:', params['save_kernel'])
        if params['save_kernel']:
            print(save_args['outdir'] + str(save_args['job_name']) + '_prop_kernel.csv')
    # Calculate propagation kernel by propagating identity matrix of network
    network_nodes = network.nodes()
    network_I = pd.DataFrame(np.identity(len(network_nodes)), index=network_nodes, columns=network_nodes)
    if params['save_kernel']:
        save_args['iteration_label'] = 'kernel'
        kernel = prop.network_propagation(network, network_I, alpha=params['prop_alpha'],
                                          symmetric_norm=params['prop_symmetric_norm'], verbose=verbose, **save_args)
    else:
        kernel = prop.network_propagation(network, network_I, alpha=params['prop_alpha'],
                                          symmetric_norm=params['prop_symmetric_norm'], verbose=verbose)
    print()
    print('##################################################################################')
    print('# Performing', params['niter'], 'iterations of NBS')
    print('##################################################################################')
    if verbose:
        print('Overall number of NBS iterations:', params['niter'])
        print('*** Subsampling parameters ***')
        print('Patient subsample rate:', params['pats_subsample_p'])
        print('Network gene subsample rate:', params['gene_subsample_p'])
        print('Minimum number of mutations allowed per subsampled patient:', params['min_muts'])
        print('*** Propagation data parameters ***')
        print('Save propagated, subsampled data (at each intermediate step):', params['save_prop'])
        print('Perform quantile normalization on propagated data:', params['qnorm_data'])
        print('*** Network-regularized NMF (netNMF) parameters ***')
        print('Number of clusters:', params['netNMF_k'])
        print('Network-regularization coefficient (netNMF lambda):', params['netNMF_lambda'])
        print('Maximum number of multiplicative updates to perform in netNMF:', params['netNMF_maxiter'])
        print('Maximum machine precision value:', params['netNMF_eps'])
        print('Maximum netNMF reconstruction error for convergence:', params['netNMF_err_tol'])
        print('Maximum change per multiplicative update step allowed in reconstruction error for convergence:', params[
            'netNMF_err_delta_tol'])
        print('Save individual H matrices to file', params['save_H'])
    # Turn off internal NBS reporting steps
    params['verbose'] = False
    # Initialize and construct Hlist
    print()
    Hlist = []
    if (params['save_prop'] == False) and (params['save_H'] == False):
        del params['outdir']
    for i in range(params['niter']):
        netNMF_time = time.time()
        # Change iteration label for each run of NBS
        if (params['save_prop']) or (params['save_H']):
            params['iteration_label'] = str(i + 1)
        # Run NBS core steps and save resulting H matrix to Hlist
        Hlist.append(
            NBS_single.NBS_single(sm_mat, knnGlap, propNet=network, propNet_kernel=kernel, k=params['netNMF_k'],
                                    **params))
        # Hlist.append(NBS_single.NBS_single(sm_mat, knnGlap, propNet=network, k=params['netNMF_k'], **params))
        # Report run time of each NBS iteration
        t = time.time() - netNMF_time
        print('NBS iteration:', i + 1, 'complete:', t, 'seconds')
    print()
    print('##################################################################################')
    print('# Performing Consensus Clustering')
    print('##################################################################################')
    if verbose:
        print('Number of consensus clusters:', params['netNMF_k'])
        print('Consensus hierarchical clustering linkage method', params['hclust_linkage_method'])
        print('Consensus hierarchical clustering linkage metric', params['hclust_linkage_metric'])
        print('Save consensus clustering results', params['save_cc_results'])
        if params['save_cc_results']:
            print(save_args['outdir'] + str(save_args['job_name']) + '_cc_matrix.csv')
            print(save_args['outdir'] + str(save_args['job_name']) + '_cluster_assignments.csv')
        print('Save patient co-clustering map', params['save_cc_map'])
        if params['save_cc_map']:
            print(save_args['outdir'] + str(save_args['job_name']) + '_cc_map.png')
    # Perform consensus clustering
    if params['save_cc_results']:
        NBS_cc_table, NBS_cc_linkage, NBS_cluster_assign = cc.consensus_hclust_hard(Hlist, k=params['netNMF_k'],
                                                                                    hclust_linkage_method=params[
                                                                                        'hclust_linkage_method'],
                                                                                    hclust_linkage_metric=params[
                                                                                        'hclust_linkage_metric'],
                                                                                    verbose=verbose, **save_args)
    else:
        NBS_cc_table, NBS_cc_linkage, NBS_cluster_assign = cc.consensus_hclust_hard(Hlist, k=params['netNMF_k'],
                                                                                    hclust_linkage_method=params[
                                                                                        'hclust_linkage_method'],
                                                                                    hclust_linkage_metric=params[
                                                                                        'hclust_linkage_metric'],
                                                                                    verbose=verbose, **save_args)
    # Save co-clustering map if desired
    if params['save_cc_map']:
        NBS_clust_cmap = plot.cluster_color_assign(NBS_cluster_assign, name='NBS Cluster')
        plot.plot_cc_map(NBS_cc_table, NBS_cc_linkage, col_color_map=NBS_clust_cmap, **save_args)
    p = 1.0
    SCM = 0.0
    # The following section only executes if patient survival data is provided
    if params['plot_survival']:
        print()
        print('##################################################################################')
        print('# Performing Survival Analysis')
        print('##################################################################################')
        if verbose:
            print('Survival data file delimiter:', repr(params['surv_file_delim']))
            print('Perform survival log-rank test:', params['surv_lr_test'])
            print('Maximum time considered in survival analysis (days):', params['surv_tmax'])
            print('Kaplan Meier Plot:')
            print(save_args['outdir'] + str(save_args['job_name']) + '_KM_plot.png')
            # Perform survival analysis via KM plot
        p, SCM = plot.cluster_KMplot(NBS_cluster_assign, args.survival_data,
                                     cancer_type='OV',
                                     delimiter=params['surv_file_delim'],
                            lr_test=params['surv_lr_test'], tmax=params['surv_tmax'], **save_args)
    else:
        print()
        print('##################################################################################')
        print('# No Survival Analysis Performed')
        print('##################################################################################')

    return p, SCM


if __name__ == "__main__":
    main()
