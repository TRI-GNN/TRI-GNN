'''
Contains classes and methods that represent topological information
about a data set.
'''

import collections.abc
import math
import itertools
import gudhi as gd
#import igraph as ig
import numpy as np
import networkx as nx


class PersistenceDiagram(collections.abc.Sequence):
    '''
    Represents a persistence diagram, i.e. a pairing of nodes in
    a graph. The purpose of this class is to provide a *simpler*
    interface for storing and accessing this pairing.
    '''

    def __init__(self):
        self._pairs = []
        self._betti = None

    def __len__(self):
        '''
        Returns the number of pairs in the persistence diagram.
        '''

        return len(self._pairs)

    def __getitem__(self, index):
        '''
        Returns the persistence pair at the given index.
        '''

        return self._pairs[index]

    def append(self, x, y, index=None):
        '''
        Appends a new persistence pair to the given diagram. Performs no
        other validity checks.

        :param x: Creation value of the given persistence pair
        :param y: Destruction value of the given persistence pair

        :param index: Optional index that helps identify a persistence
        pair using information stored *outside* the diagram.
        '''

        self._pairs.append((x, y, index))

    def total_persistence(self, p=1):
        '''
        Calculates the total persistence of the current pairing.
        '''

        return sum([abs(x - y)**p for x, y, _ in self._pairs])**(1.0 / p)

    def infinity_norm(self, p=1):
        '''
        Calculates the infinity norm of the current pairing.
        '''

        return max([abs(x - y)**p for x, y, _ in self._pairs])

    def remove_diagonal(self):
        '''
        Removes diagonal elements, i.e. elements for which x and
        y coincide.
        '''

        self._pairs = [(x, y, c) for x, y, c in self._pairs if x != y]

    @property
    def betti(self):
        '''
        :return: Betti number of the current persistence diagram or
        `None` if no number has been assigned.
        '''

        return self._betti

    @betti.setter
    def betti(self, value):
        '''
        Sets the Betti number of the current persistence diagram.

        :param value: Betti number to assign. The function will perform
        a brief consistency check by counting the number of persistence
        pairs.
        '''

        if value > len(self):
            raise RuntimeError(
                '''
                Betti number must be less than or equal to persistence
                diagram cardinality
                '''
            )

        self._betti = value

    def __repr__(self):
        '''
        :return: String-based representation of the diagram
        '''

        return '\n'.join([f'{x} {y} [{c}]' for x, y, c in self._pairs])


class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, num_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = [x for x in range(num_vertices)]

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistenceDiagramCalculator:
    '''
    Given a weighted graph, calculates a persistence diagram. The client
    can modify the filtration order and the vertex weight assignment.
    '''

    def __init__(self,
                 order='sublevel',
                 unpaired_value=None,
                 vertex_attribute=None):
        '''
        Initializes a new instance of the persistence diagram
        calculation class.

        :param order: Filtration order (ignored for now)
        :param unpaired_value: Value to use for unpaired vertices. If
        not specified the largest weight (sublevel set filtration) is
        used.
        :param vertex_attribute: Graph attribute to query for vertex
        values. If not specified, no vertex attributes will be used,
        and each vertex will be assigned a value of zero.
        '''

        self._order = order
        self._unpaired_value = unpaired_value
        self._vertex_attribute = vertex_attribute

        if self._order not in ['sublevel', 'superlevel']:
            raise RuntimeError(
                '''
                Unknown filtration order \"{}\"
                '''.format(self._order)
            )

    def fit_transform(self, graph):
        '''
        Applies a filtration to a graph and calculates its persistence
        diagram. The function will return the persistence diagram plus
        all edges that are involved in cycles.

        :param graph: Weighted graph whose persistence diagram will be
        calculated.

        :return: Tuple consisting of the persistence diagram, followed
        by a list of all edge indices that create a cycle.
        '''

        num_vertices = graph.vcount()
        uf = UnionFind(num_vertices)

        edge_weights = np.array(graph.es['weight'])   # All edge weights
        edge_indices = None                           # Ordering for filtration
        edge_indices_cycles = []                      # Edge indices of cycles

        if self._order == 'sublevel':
            edge_indices = np.argsort(edge_weights, kind='stable')
        elif self._order == 'superlevel':
            edge_indices = np.argsort(-edge_weights, kind='stable')

        assert edge_indices is not None

        # Will be filled during the iteration below. This will become
        # the return value of the function.
        pd = PersistenceDiagram()

        # Go over all edges and optionally create new points for the
        # persistence diagram.
        for edge_index, edge_weight in zip(edge_indices, edge_weights[edge_indices]):
            u, v = graph.es[edge_index].tuple

            # Preliminary assignment of younger and older component. We
            # will check below whether this is actually correct, for it
            # is possible that u is actually the older one.
            younger = uf.find(u)
            older = uf.find(v)

            # Nothing to do here: the two components are already the
            # same
            if younger == older:
                edge_indices_cycles.append(edge_index)
                continue

            # Ensures that the older component precedes the younger one
            # in terms of its vertex index
            elif younger > older:
                u, v = v, u
                younger, older = older, younger

            vertex_weight = 0.0

            # Vertex attributes have been set, so we use them for the
            # persistence diagram creation below.
            if self._vertex_attribute:
                vertex_weight = graph.vs[self._vertex_attribute][younger]

            creation = vertex_weight    # x coordinate for persistence diagram
            destruction = edge_weight   # y coordinate for persistence diagram

            uf.merge(u, v)
            pd.append(creation, destruction, younger)

        # By default, use the largest (sublevel set) or lowest
        # (superlevel set) weight, unless the user specified a
        # different one.
        unpaired_value = edge_weights[edge_indices[-1]]
        if self._unpaired_value:
            unpaired_value = self._unpaired_value

        # Add tuples for every root component in the Union--Find data
        # structure. This ensures that multiple connected components
        # are handled correctly.
        for root in uf.roots():

            vertex_weight = 0.0

            # Vertex attributes have been set, so we use them for the
            # creation of the root tuple.
            if self._vertex_attribute:
                vertex_weight = graph.vs[self._vertex_attribute][root]

            creation = vertex_weight
            destruction = unpaired_value

            pd.append(creation, destruction, root)

            if pd.betti is not None:
                pd.betti = pd.betti + 1
            else:
                pd.betti = 1

        return pd, edge_indices_cycles



def WL_attributes(adj, features,iteration):
    WL_features = np.zeros(shape=(features.shape), dtype= np.float32)
    deg_inverse = (1./np.sum(adj, axis= 1)).reshape((1,-1))
    for i in range(iteration):
        if i == 0:
            WL_features = 0.5 * (features + np.transpose(deg_inverse * np.transpose(np.matmul(adj, features))))
        else:
            WL_features = 0.5 * (WL_features + np.transpose(deg_inverse * np.transpose(np.matmul(adj, WL_features))))
    return WL_features


def WL_w_adj_attributes(adj, w_adj, features, iteration):
    WL_w_adj_features = np.zeros(shape=(features.shape), dtype= np.float32)
    deg_inverse = (1./np.sum(adj, axis= 1)).reshape((1,-1))
    for i in range(iteration):
        print(i)
        if i == 0:
            WL_w_adj_features = 0.5 * (features + np.transpose(deg_inverse * np.transpose(np.matmul(w_adj, features))))
        else:
            WL_w_adj_features = 0.5 * (WL_w_adj_features + np.transpose(deg_inverse * np.transpose(np.matmul(w_adj, WL_w_adj_features))))
    return WL_w_adj_features


def WL_attributes_distance_mat(WL_features):
    WL_distance_mat = np.zeros(shape=(WL_features.shape[0], WL_features.shape[0]),dtype= np.float32)
    for i in range(WL_features.shape[0]-1):
        print(i)
        for j in range(i+1, WL_features.shape[0]):
            WL_distance_mat[i,j] = np.sqrt(np.sum(np.square(WL_features[i,:] - WL_features[j,:])))
            WL_distance_mat[j,i] = np.sqrt(np.sum(np.square(WL_features[i,:] - WL_features[j,:])))
    return WL_distance_mat


def edge_weight_func(features_array):
    edge_weight_mat = np.zeros(shape = (features_array.shape[0],features_array.shape[0]), dtype= np.float32)
    for i in range(features_array.shape[0]-1):
        print(i)
        for j in range(i+1, features_array.shape[0]):
            sources = features_array[i, :]
            targets = features_array[j, :]
            intersection = np.logical_and(sources, targets)
            union = np.logical_or(sources, targets)

            if intersection.sum() / union.sum() == 1.:
                edge_weight_mat[i, j] = 1e-3
                edge_weight_mat[j, i] = 1e-3
            else:
                edge_weight_mat[i, j] = 1. - intersection.sum() / union.sum()
                edge_weight_mat[j, i] = 1. - intersection.sum() / union.sum()
    return edge_weight_mat



def k_th_order_subgraph(adj_mat, k):
    output = list()
    G = nx.from_numpy_matrix(adj_mat)
    for i in range(adj_mat.shape[0]):
        v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=k).items()]
        tmp_subgraph = adj_mat[np.ix_(v_labels, v_labels)]
        output.append(tmp_subgraph)
    return output


def k_th_order_subgraph_features(adj_mat, features, k):
    output = list()
    G = nx.from_numpy_matrix(adj_mat)
    for i in range(adj_mat.shape[0]):
        v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=k).items()]
        tmp_subgraph_features = features[v_labels,:]
        output.append(tmp_subgraph_features)
    return output


def k_th_order_weighted_subgraph(adj_mat, w_adj_mat, k):
    output = list()
    G = nx.from_numpy_matrix(adj_mat)
    for i in range(adj_mat.shape[0]):
        v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=k).items()]
        tmp_subgraph = w_adj_mat[np.ix_(v_labels, v_labels)]

        #tmp_subgraph = np.where(tmp_subgraph == 1., 1e-3, 1.-tmp_subgraph)
        #np.fill_diagonal(tmp_subgraph, 0.)

        output.append(tmp_subgraph)
    return output


def first_order_subgraph(adj_mat):
    output = list()
    for i in range(adj_mat.shape[0]):
        tmp_neighbors = np.where(adj_mat[i,:]>0)
        tmp_neighbors = tmp_neighbors[0].tolist()
        if i not in tmp_neighbors:
            tmp_neighbors.append(i)
            tmp_neighbors = sorted(tmp_neighbors)
        tmp_subgraph = adj_mat[np.ix_(tmp_neighbors, tmp_neighbors)]
        output.append(tmp_subgraph)
    return output


def second_order_subgraph(adj_mat):
    output = list()
    for i in range(adj_mat.shape[0]):
        tmp_first_neighbors = np.where(adj_mat[i,:]>0)
        tmp_first_neighbors = tmp_first_neighbors[0].tolist()

        tmp_second_neighbors_output = list()
        for j in range(len(tmp_first_neighbors)):
            tmp_second_neighbors = np.where(adj_mat[tmp_first_neighbors[j], :] > 0)
            tmp_second_neighbors = tmp_second_neighbors[0].tolist()
            tmp_second_neighbors_output = tmp_second_neighbors_output + tmp_second_neighbors

        tmp_final_neighbors = tmp_first_neighbors + tmp_second_neighbors_output
        if i not in tmp_final_neighbors:
            tmp_final_neighbors.append(i)
        tmp_final_neighbors = sorted(tmp_final_neighbors)
        tmp_final_neighbors = np.unique(tmp_final_neighbors).tolist()

        tmp_final_subgraph = adj_mat[np.ix_(tmp_final_neighbors, tmp_final_neighbors)]
        output.append(tmp_final_subgraph)
    return output


def third_order_subgraph(adj_mat):
    output = list()
    for i in range(adj_mat.shape[0]):
        tmp_first_neighbors = np.where(adj_mat[i,:]>0)
        tmp_first_neighbors = tmp_first_neighbors[0].tolist()

        tmp_second_neighbors_output = list()
        for j in range(len(tmp_first_neighbors)):
            tmp_second_neighbors = np.where(adj_mat[tmp_first_neighbors[j], :] > 0)
            tmp_second_neighbors = tmp_second_neighbors[0].tolist()
            tmp_second_neighbors_output = tmp_second_neighbors_output + tmp_second_neighbors
        tmp_second_neighbors_output = np.unique(tmp_second_neighbors_output).tolist()


        tmp_third_neighbors_output = list()
        for k in range(len(tmp_second_neighbors_output)):
            tmp_third_neighbors = np.where(adj_mat[tmp_second_neighbors_output[k], :] > 0)
            tmp_third_neighbors = tmp_third_neighbors[0].tolist()
            tmp_third_neighbors_output = tmp_third_neighbors_output + tmp_third_neighbors
        tmp_third_neighbors_output = np.unique(tmp_third_neighbors_output).tolist()


        tmp_final_neighbors = tmp_first_neighbors + tmp_second_neighbors_output + tmp_third_neighbors_output
        if i not in tmp_final_neighbors:
            tmp_final_neighbors.append(i)
        tmp_final_neighbors = sorted(tmp_final_neighbors)
        tmp_final_neighbors = np.unique(tmp_final_neighbors).tolist()
        if i == 100:
            print(tmp_final_neighbors)

        tmp_final_subgraph = adj_mat[np.ix_(tmp_final_neighbors, tmp_final_neighbors)]
        output.append(tmp_final_subgraph)
    return output



def fourth_order_subgraph(adj_mat):
    output = list()
    for i in range(adj_mat.shape[0]):
        tmp_first_neighbors = np.where(adj_mat[i,:]>0)
        tmp_first_neighbors = tmp_first_neighbors[0].tolist()

        tmp_second_neighbors_output = list()
        for j in range(len(tmp_first_neighbors)):
            tmp_second_neighbors = np.where(adj_mat[tmp_first_neighbors[j], :] > 0)
            tmp_second_neighbors = tmp_second_neighbors[0].tolist()
            tmp_second_neighbors_output = tmp_second_neighbors_output + tmp_second_neighbors
        tmp_second_neighbors_output = np.unique(tmp_second_neighbors_output).tolist()


        tmp_third_neighbors_output = list()
        for k in range(len(tmp_second_neighbors_output)):
            tmp_third_neighbors = np.where(adj_mat[tmp_second_neighbors_output[k], :] > 0)
            tmp_third_neighbors = tmp_third_neighbors[0].tolist()
            tmp_third_neighbors_output = tmp_third_neighbors_output + tmp_third_neighbors
        tmp_third_neighbors_output = np.unique(tmp_third_neighbors_output).tolist()


        tmp_fourth_neighbors_output = list()
        for q in range(len(tmp_third_neighbors_output)):
            tmp_fourth_neighbors = np.where(adj_mat[tmp_third_neighbors_output[q], :] > 0)
            tmp_fourth_neighbors = tmp_fourth_neighbors[0].tolist()
            tmp_fourth_neighbors_output = tmp_fourth_neighbors_output + tmp_fourth_neighbors
        tmp_fourth_neighbors_output = np.unique(tmp_fourth_neighbors_output).tolist()


        tmp_final_neighbors = tmp_first_neighbors + tmp_second_neighbors_output + tmp_third_neighbors_output + tmp_fourth_neighbors_output
        if i not in tmp_final_neighbors:
            tmp_final_neighbors.append(i)
        tmp_final_neighbors = sorted(tmp_final_neighbors)
        tmp_final_neighbors = np.unique(tmp_final_neighbors).tolist()

        tmp_final_subgraph = adj_mat[np.ix_(tmp_final_neighbors, tmp_final_neighbors)]
        output.append(tmp_final_subgraph)
    return output


def second_order_weighted_subgraph(adj_mat, weighted_adj_mat):
    output = list()
    for i in range(adj_mat.shape[0]):
        tmp_first_neighbors = np.where(adj_mat[i,:]>0)
        tmp_first_neighbors = tmp_first_neighbors[0].tolist()

        tmp_second_neighbors_output = list()
        for j in range(len(tmp_first_neighbors)):
            tmp_second_neighbors = np.where(adj_mat[tmp_first_neighbors[j], :] > 0)
            tmp_second_neighbors = tmp_second_neighbors[0].tolist()
            tmp_second_neighbors_output = tmp_second_neighbors_output + tmp_second_neighbors

        tmp_final_neighbors = tmp_first_neighbors + tmp_second_neighbors_output
        if i not in tmp_final_neighbors:
            tmp_final_neighbors.append(i)
        tmp_final_neighbors = sorted(tmp_final_neighbors)
        tmp_final_neighbors = np.unique(tmp_final_neighbors).tolist()

        tmp_label_adj = np.where(adj_mat[np.ix_(tmp_final_neighbors, tmp_final_neighbors)]>0)
        tmp_label_weight_adj = np.where(weighted_adj_mat[np.ix_(tmp_final_neighbors, tmp_final_neighbors)] > 0)
        if np.array_equal(tmp_label_adj[0], tmp_label_weight_adj[0]) & np.array_equal(tmp_label_adj[1], tmp_label_weight_adj[1]):
            tmp_final_subgraph = weighted_adj_mat[np.ix_(tmp_final_neighbors, tmp_final_neighbors)]
        else:
            tmp_final_subgraph = adj_mat[np.ix_(tmp_final_neighbors, tmp_final_neighbors)] * 1e-5
        output.append(tmp_final_subgraph)
    return output


def assign_filtration_values(
        graph,
        attributes,
        order='sublevel',
        normalize=False):
    '''
    Given a vertex attribute of a graph, assigns filtration values as
    edge weights to the graph edges.

    :param graph: Graph to modify
    :param attribute: Attribute sequence to use for the filtration
    :param order: Order of filtration
    :param normalize: If set, normalizes according to filtration order

    :return: Graph with added edges
    '''

    selection_function = max if order == 'sublevel' else min

    if normalize:
        offset = np.max(attributes) if order == 'sublevel' \
                                    else np.min(attributes)
    else:
        offset = 1.0

    for edge in graph.es:
        source = edge.source
        target = edge.target

        source_weight = attributes[source] / offset
        target_weight = attributes[target] / offset
        edge_weight = selection_function(source_weight, target_weight)

        edge['weight'] = edge_weight

    return graph


def func_weight(bd, arc_c, arc_p):
    return np.maximum(
        np.arctan(math.pow((bd[1] - bd[0]) / arc_c, arc_p)), 0.0)


def vector_weight(diagram):
    num_point = int(diagram.size / 2)
    vec = np.empty(num_point)
    for k in range(num_point):
        vec[k] = func_weight(diagram[k, :],arc_c=1., arc_p=5.0) #5.0
    return vec


def func_kernel(bd_1, bd_2, sigma=1.0):
    dif_vec = bd_1 - bd_2
    squared_distance = dif_vec[0] ** 2 + dif_vec[1] ** 2
    return np.exp(-1.0 * squared_distance / (2.0 * math.pow(sigma, 2)))


def kernel_linear(diagram_1, diagram_2, vec_weight_1, vec_weight_2):
    s = 0.0
    num_point_1 = int(diagram_1.size / 2)
    num_point_2 = int(diagram_2.size / 2)
    for i in range(num_point_1):
        for j in range(num_point_2):
            s += (vec_weight_1[i] * vec_weight_2[j]
                  * func_kernel(diagram_1[i, :], diagram_2[j, :]))
    return s


def simplicial_complex_dgm(target_matrix):
    rips_complex = gd.RipsComplex(distance_matrix=target_matrix, max_edge_length=1.)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    res = np.zeros(shape=(np.array(diag)[:,1].shape[0],2), dtype= np.float32)
    for i in range(np.array(diag).shape[0]):
        if np.isfinite(np.array(diag)[:,1][i][1]):
            res[i, :] = np.array(diag)[:, 1][i]
    return res


def exp_weighting_func(dgm, constant, sigma):
    birth_time = dgm[:, 0]
    death_time = dgm[:, 1]
    tmp = np.arctan(constant*np.power(death_time - birth_time,2.))* np.exp(-(death_time+birth_time)/(sigma**2))
    return np.sum(tmp)
