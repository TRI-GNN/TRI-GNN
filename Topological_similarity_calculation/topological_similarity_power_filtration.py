import numpy as np
from utils import *
from topology import *
import gudhi as gd
import networkx as nx
from persim import wasserstein
import pandas as pd
from topology import edge_weight_func


# load data, e.g., cora
dataset = 'cora'
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, label = load_data(dataset)
adj_array = adj.toarray().astype(np.float32)
# power filtraion, i.e., topological distance


# To get node_similarity_matrix.npy, please run edge_weight_func on node feature matrix
node_similarity_matrix = edge_weight_func(features.toarray())
k_hop = 3
k_hop_subgraphs = k_th_order_weighted_subgraph(adj_mat = adj_array, w_adj_mat= node_similarity_matrix, k = k_hop)


power_filtration_dgms = list()
for i in range(len(k_hop_subgraphs)):
    print(i)
    power_filtration_dgm = simplicial_complex_dgm(k_hop_subgraphs[i])
    if power_filtration_dgm.size == 0:
        power_filtration_dgms.append(np.array([]))
    else:
        power_filtration_dgms.append(power_filtration_dgm)

power_dgms = np.array(power_filtration_dgms)


# for any pair of nodes
# wasserstein distances calculation
wasserstein_distances = np.zeros(adj_array.shape, dtype = np.float32)
for i in range(adj_array.shape[0] - 1):
    for j in range(i+1, adj_array.shape[0]):
        wasserstein_distances[i, j] = wasserstein(power_dgms[i], power_dgms[j])
        wasserstein_distances[j, i] = wasserstein(power_dgms[j], power_dgms[i]) # wasserstein distance is not symmetric

np.save(dataset + '_power_filtration_wasserstein_distances', wasserstein_distances)


# for k-hop neighborhood
# wasserstein distances calculation
k = 3 # only consider calculating the distances between node u and its k-hop neighborhood
G = nx.from_numpy_matrix(adj_array)
wasserstein_distances = np.zeros(adj_array.shape, dtype = np.float32)
for i in range(adj_array.shape[0] - 1):
    v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=k).items()]
    for j in v_labels:
        wasserstein_distances[i, j] = wasserstein(power_dgms[i], power_dgms[j])
        wasserstein_distances[j, i] = wasserstein(power_dgms[j], power_dgms[i]) # wasserstein distance is not symmetric

np.save(dataset + '_' +  str(k) + '_hop_'+ 'power_filtration_wasserstein_distances', wasserstein_distances)

