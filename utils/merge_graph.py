import numpy as np
import pandas as pd
import torch
import data_import_tools as dit
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize


def construct_graph(features, method='heat', topk=11, prop_data=False):
    dist = None
    if prop_data is False:
        if method == 'heat':
            dist = -0.5 * pair(features) ** 2
            dist = np.exp(dist)
        elif method == 'cos':
            features[features > 0] = 1
            dist = np.dot(features, features.T)
        elif method == 'ncos':
            features[features > 0] = 1
            features = normalize(features, axis=1, norm='l1')
            dist = np.dot(features, features.T)

        elif method == 'p':
            y = features.T - np.mean(features.T)
            features = features - np.mean(features)
            dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    # prop_data is patient-patient adjacent matrix already
    else:
        # features[features > 0.5] = 1
        # features = normalize(features, axis=1, norm='l1')
        dist = np.array(features)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    # A = np.zeros_like(dist)
    edges = []
    for i, v in enumerate(inds):
        # mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                edges.append('{}\t{}'.format(i, vv))
    return edges


# merge two gene node graph, to add gene edge in one graph into another gene node graph
# mutation_file is patient * mutation (eg:OV 328*9850)
def merge_gene_graph(mutation_file, gene_graph_file, merged_graph_file):
    binary_mat = pd.read_csv(mutation_file, delimiter=',', index_col=0).astype(int)
    genes = np.array(binary_mat.columns, dtype=int)
    # patients = np.array(binary_mat.index)
    binary_mat = np.array(binary_mat)
    gene_select = []  # select mutation genes(value==1)
    for row in binary_mat:
        gene_select.append(np.sort(genes[np.where(row == 1)]))
    edges = []
    for i in gene_select:
        for j in range(len(i)):
            for k in range(j + 1, len(i)):
                edges.append(str(i[j]) + '\t' + str(i[k]))
    edges = set(edges)

    # choose part1 to merge mut_graph and PPI network,choose part2 to make and save a mut_graph only
    # ----------- part1-----------------------
    # gene_graph = []
    # with open(gene_graph_file, 'r') as f:
    #     for edge in f.readlines():
    #         edge = edge.split('\n')[0]
    #         gene_graph.append(edge)
    # gene_graph = set(gene_graph)
    # merged_graph = gene_graph.union(edges)
    #
    # f1 = open(merged_graph_file + 'gene_merged_graph.txt', 'w')
    # for i in merged_graph:
    #     f1.write(i + '\n')
    # f1.close()
    # ----------- part2-----------------------
    f1 = open(merged_graph_file, 'w')
    for i in edges:
        f1.write(i + '\n')
    f1.close()


# merge two gene node graph, to add gene edge in one graph into another gene node graph
# requirement:for two nodes contain in mut_edge must also in HM90
#             for two nodes contain in HM90_edge must also in mut genes(eg: OV has 9850 mut genes)
def merge_gene_graph_2(mutation_file, gene_graph_file, merged_graph_file):
    binary_mat = pd.read_csv(mutation_file, delimiter=',', index_col=0).astype(int)
    mut_genes = np.array(binary_mat.columns, dtype=int)  # all mut genes
    # patients = np.array(binary_mat.index)
    binary_mat = np.array(binary_mat)
    gene_select = []  # select mutation genes(value==1) the shape is N*Xi
    for row in binary_mat:
        gene_select.append(np.sort(mut_genes[np.where(row == 1)]))

    net_edges = []  # all net edges in network
    with open(gene_graph_file, 'r') as f:
        for edge in f.readlines():
            edge = edge.split('\n')[0]
            net_edges.append(edge)

    start_nodes, end_nodes = [], []
    for item in net_edges:
        start_nodes.append(item.split('\t')[0])
        end_nodes.append(item.split('\t')[1])
    net_genes = set(start_nodes).union(set(end_nodes))

    # filt mut gene,just save which contained in interaction network
    # save meet requirements mut edges
    mut_edges = []
    for i in gene_select:
        for j in range(len(i)):
            for k in range(j + 1, len(i)):
                if str(i[j]) in net_genes and str(i[k]) in net_genes:
                    mut_edges.append(str(i[j]) + '\t' + str(i[k]))

    # filt net gene,just save which contained in all relevant cancer genes(OV:9850)
    for i in range(len(net_edges))[::-1]:
        if start_nodes[i] not in mut_genes or end_nodes[i] not in mut_genes:
            net_edges.pop(i)

    # merge two graph
    mut_edges = set(mut_edges)
    net_edges = set(net_edges)
    merged_graph = net_edges.union(mut_edges)

    f1 = open(merged_graph_file, 'w')
    for i in merged_graph:
        f1.write(i + '\n')
    f1.close()


# keep node num stay when merging graph
# merge two gene node graph, to add part most *% frequent gene edge in one graph into another gene node graph
# mutation_file is patient * mutation (eg:OV 328*9850)
def merge_part_gene_graph(mutation_file, gene_graph_file, merged_graph_file, part_rate=0.1, keep_nodes=True):
    binary_mat = pd.read_csv(mutation_file, delimiter=',', index_col=0).astype(int)
    genes = np.array(binary_mat.columns, dtype=int)
    # patients = np.array(binary_mat.index)
    binary_mat = np.array(binary_mat)
    gene_select = []  # select mutation genes(value==1)
    for row in binary_mat:
        gene_select.append(np.sort(genes[np.where(row == 1)]))

    # fitter selected genes only which belongs network
    # to keep network node num,just add edges between these have existed nodes
    network = dit.load_network_file(gene_graph_file, delimiter='\t',
                                    degree_shuffle=False,
                                    label_shuffle=False, verbose=True)
    nodes = np.array(network.nodes).astype(int)
    fitter = []
    for row in gene_select:
        item = []
        for gene in row:
            if keep_nodes:
                if gene in nodes:
                    item.append(gene)
            else:
                item.append(gene)
        fitter.append(np.array(item))

    edges = []
    for i in fitter:
        for j in range(len(i)):
            for k in range(j + 1, len(i)):
                edges.append(str(i[j]) + ',' + str(i[k]))

    # choose edges without repeat, the edge num is same as set() way, but the result in set() is disordered
    # edge_set1 = set(edges)
    edge_set = dict()
    for edge in edges:
        if edge not in edge_set:
            edge_set[edge] = 1
        else:
            edge_set[edge] += 1

    # sort choosed edges in desend frequence, and then choose edges in part_rate further
    edge_tuplelist = list(zip(edge_set.values(), edge_set.keys()))
    edge_tuplelist_sorted = sorted(edge_tuplelist, reverse=True)
    merge_edges_num = int(len(edge_set) * part_rate)
    select_edges = []
    for item in range(merge_edges_num):
        select_edges.append(edge_tuplelist_sorted[item][1])
    select_edges = set(select_edges)

    # loading gene graph to merge with final selected edges
    gene_graph = []
    with open(gene_graph_file, 'r') as f:
        for edge in f.readlines():
            edge = edge.split('\n')[0]
            gene_graph.append(edge)
    gene_graph = set(gene_graph)
    merged_graph = gene_graph.union(select_edges)

    # save the merged gene graph
    if keep_nodes:
        f1 = open(merged_graph_file + 'keep_nodes_{:.2f}_part_genes_merged_graph_1.txt'.format(part_rate), 'w')
    else:
        f1 = open(merged_graph_file + 'nodes_added_{:.2f}_part_genes_merged_graph.txt'.format(part_rate), 'w')
    for i in merged_graph:
        f1.write(i + '\n')
    f1.close()


# merge two patient node graph,the one is organized from network propagating data(eg: ov 328*4),
# another is constructed with original mutation data(eg: OV 328*9850)
def merge_patient_graph(propgrated_data_file, mutation_file, merged_graph_file):
    topk = 11
    method = ['heat', 'cos', 'ncos', 'p']
    mutation_features = (pd.read_csv(mutation_file, index_col=0)).astype(float)
    propagated_adj = (pd.read_csv(propgrated_data_file, index_col=0)).astype(float)

    mutation_graph = set(construct_graph(mutation_features, method=method[2], topk=topk, prop_data=False))
    propagated_graph = set(construct_graph(propagated_adj, method=method[2], topk=topk, prop_data=True))
    merged_graph = mutation_graph.union(propagated_graph)

    # save the merged gene graph
    f = open(merged_graph_file + 'top' + str(topk) + '_patient_merged_graph.txt', 'w')
    for i in merged_graph:
        f.write(i + '\n')
    f.close()


if __name__ == "__main__":
    mutation_file = ''
    gene_graph_file = ''
    merged_graph_file = ''
    merge_gene_graph(mutation_file=mutation_file, gene_graph_file=gene_graph_file,
                     merged_graph_file=merged_graph_file)
