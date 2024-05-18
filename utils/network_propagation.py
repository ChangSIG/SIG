import networkx as nx
import time
import numpy as np
import pandas as pd


def normalize_network(network, symmetric_norm=False):
    adj_mat = nx.adjacency_matrix(network)
    adj_array = np.array(adj_mat.todense())
    if symmetric_norm:
        D = np.diag(1 / np.sqrt(sum(adj_array)))
        adj_array_norm = np.dot(np.dot(D, adj_array), D)
    else:
        degree = sum(adj_array)
        adj_array_norm = (adj_array * 1.0 / degree).T
    return adj_array_norm


def fast_random_walk(alpha, binary_mat, subgraph_norm, prop_data_prev):
    term1 = (1 - alpha) * binary_mat
    term2 = np.identity(binary_mat.shape[1]) - alpha * subgraph_norm
    term2_inv = np.linalg.inv(term2)
    subgraph_prop = np.dot(term1, term2_inv)
    prop_data_add = np.concatenate((prop_data_prev, subgraph_prop), axis=1)
    return prop_data_add


def network_propagation(network, binary_matrix, alpha=0.7, symmetric_norm=False, verbose=True, **save_args):
    alpha = float(alpha)
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError('Alpha must be a value between 0 and 1')
    starttime = time.time()
    if verbose:
        print('Performing network propagation with alpha:', alpha)
    subgraphs = list(network.subgraph(c) for c in nx.connected_components(network))
    subgraph = subgraphs[0]
    subgraph_nodes = list(subgraph.nodes)
    prop_data_node_order = list(subgraph_nodes)
    binary_matrix_filt = np.array(binary_matrix.T.loc[subgraph_nodes].fillna(0).T)
    subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
    prop_data_empty = np.zeros((binary_matrix_filt.shape[0], 1))
    prop_data = fast_random_walk(alpha, binary_matrix_filt, subgraph_norm, prop_data_empty)
    for subgraph in subgraphs[1:]:
        subgraph_nodes = list(subgraph.nodes)
        prop_data_node_order = prop_data_node_order + subgraph_nodes
        binary_matrix_filt = np.array(binary_matrix.T.loc[subgraph_nodes].fillna(0).T)
        subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
        prop_data = fast_random_walk(alpha, binary_matrix_filt, subgraph_norm, prop_data)
    prop_data_df = pd.DataFrame(data=prop_data[:, 1:], index=binary_matrix.index, columns=prop_data_node_order)
    if 'outdir' in save_args:
        if 'job_name' in save_args:
            if 'iteration_label' in save_args:
                save_path = save_args['outdir'] + str(save_args['job_name']) + '_prop_' + str(
                    save_args['iteration_label']) + '.csv'
            else:
                save_path = save_args['outdir'] + str(save_args['job_name']) + '_prop.csv'
        else:
            if 'iteration_label' in save_args:
                save_path = save_args['outdir'] + 'prop_' + str(save_args['iteration_label']) + '.csv'
            else:
                save_path = save_args['outdir'] + 'prop.csv'
        prop_data_df.to_csv(save_path)
        if verbose:
            print('Network Propagation Result Saved:', save_path)
    else:
        pass
    if verbose:
        print('Network Propagation Complete:', time.time() - starttime, 'seconds')
    return prop_data_df


def network_kernel_propagation(network, network_kernel, binary_matrix, verbose=False, **save_args):
    starttime = time.time()
    if verbose:
        print('Performing network propagation with network kernel')
    subgraph_nodelists = list(nx.connected_components(network))
    prop_nodelist = list(subgraph_nodelists[0])
    prop_data = np.dot(binary_matrix.T.loc[prop_nodelist].fillna(0).T,
                       network_kernel.loc[prop_nodelist][prop_nodelist])
    for nodelist in subgraph_nodelists[1:]:
        subgraph_nodes = list(nodelist)
        prop_nodelist = prop_nodelist + subgraph_nodes
        subgraph_prop_data = np.dot(binary_matrix.T.loc[subgraph_nodes].fillna(0).T,
                                    network_kernel.loc[subgraph_nodes][subgraph_nodes])
        prop_data = np.concatenate((prop_data, subgraph_prop_data), axis=1)
    prop_data_df = pd.DataFrame(data=prop_data, index=binary_matrix.index, columns=prop_nodelist)
    if 'outdir' in save_args:
        if 'job_name' in save_args:
            if 'iteration_label' in save_args:
                save_path = save_args['outdir'] + str(save_args['job_name']) + '_prop_' + str(
                    save_args['iteration_label']) + '.csv'
            else:
                save_path = save_args['outdir'] + str(save_args['job_name']) + '_prop.csv'
        else:
            if 'iteration_label' in save_args:
                save_path = save_args['outdir'] + 'prop_' + str(save_args['iteration_label']) + '.csv'
            else:
                save_path = save_args['outdir'] + 'prop.csv'
        prop_data_df.to_csv(save_path)
    else:
        pass
    if verbose:
        print('Network Propagation Complete:', time.time() - starttime, 'seconds')
    return prop_data_df
