# A Python program for Prim's Minimum Spanning Tree (MST) algorithm.
# record the trace for a selected number of steps or vertexes
import pandas as pd
import numpy as np
import sys  # Library for INT_MAX
import igraph as ig
from numpy.linalg import norm
from scipy.stats import pearsonr
from tqdm import tqdm


class MaxSpanningGraph:

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

        # A utility function to print the constructed MST stored in parent[]

    def print_mst(self, parent, mst_set, label_list):
        print("Edge \tWeight")
        for i in range(1, self.V):
            if mst_set[i] == True:
                print(label_list[parent[i]], "-", label_list[i], "\t", round(self.graph[i][parent[i]], 2))

    def save_mst(self, parent, mst_set, label_list):
        edges = []
        edge_weights = []
        for i in range(1, self.V):
            if mst_set[i] == True:
                edges.append((label_list[parent[i]], label_list[i]))
                edge_weights.append(round(self.graph[i][parent[i]], 2))
        return edges, edge_weights

    # maximum distance value, from the set of vertices
    # not yet included in shortest path tree
    def max_key(self, key, mstSet):
        # Initilaize max value
        max_value = float('-inf')
        for v in range(self.V):
            if key[v] > max_value and mstSet[v] == False:
                max_value = key[v]
                max_index = v
        return max_index
        # Function to construct and print MST for a graph

    # represented using adjacency matrix representation
    def prim_mst(self, steps, label_list):
        """
        :param steps: the number of steps to be executed
        :return:
        """
        # Key values used to pick maximum weight edge in cut
        key = [float('-inf')] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mst_set = [False] * self.V
        parent[0] = -1  # First node is always the root of
        for cout in tqdm(range(steps)):
            # Pick the maximum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.max_key(key, mst_set)
            # Put the minimum distance vertex in
            # the shortest path tree
            mst_set[u] = True
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the longest path tree
            for v in range(self.V):
                # graph[u][v] is non zero only for adjacent vertices of m
                # mst_set[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > key[v] and mst_set[v] == False:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        self.print_mst(parent, mst_set, label_list)
        return self.save_mst(parent, mst_set, label_list), mst_set


def get_co_score(list_name, similarity):
    """
    get the co_score
    :param list_name:
    :param similarity: the method to calculate the similarity
    :return: len(list)*len(list) similarity matrix
    """
    toxic_comments_labels = np.array(list_name)
    co_count_matrix = np.zeros((toxic_comments_labels.shape[1], toxic_comments_labels.shape[1]))
    co_score = np.zeros_like(co_count_matrix)
    if similarity == "default":
        # get each label's count
        count_list = []
        for i in range(toxic_comments_labels.shape[1]):
            count_list.append(np.sum(toxic_comments_labels[:, i]))

        # get co_occurrence count
        for i in range(co_count_matrix.shape[0]):
            for j in range(i + 1, co_count_matrix.shape[1]):
                co_count_matrix[i, j] = np.sum(toxic_comments_labels[:, i] * toxic_comments_labels[:, j])

        # get co_score
        for i in range(co_score.shape[0]):
            for j in range(i + 1, co_score.shape[1]):
                co_score[i, j] = co_count_matrix[i, j] / np.minimum(count_list[i], count_list[j])
                co_score[j, i] = co_score[i, j]
    elif similarity == "cosine":
        for i in range(co_score.shape[0]):
            for j in range(i + 1, co_score.shape[1]):
                a = toxic_comments_labels[:, i]
                b = toxic_comments_labels[:, j]
                co_score[i, j] = np.inner(a, b)/(norm(a)*norm(b))
                co_score[j, i] = co_score[i, j]
    elif similarity == "pearson":
        for i in range(co_score.shape[0]):
            for j in range(i + 1, co_score.shape[1]):
                a = toxic_comments_labels[:, i]
                b = toxic_comments_labels[:, j]
                co_score[i, j], _ = np.abs(pearsonr(a, b))
                co_score[j, i] = co_score[i, j]
    return co_score


def edges_selected(threshold, edges, weights):
    """
    select the edge the weight of which is bigger than threshold
    :param threshold:
    :param edges:
    :param weights:
    :return:
    """
    edges_new = []
    weights_new = []
    for index in range(len(weights)):
        if weights[index] >= threshold:
            edges_new.append(edges[index])
            weights_new.append(weights[index])
    unique_v = set(sum(edges_new, ()))
    return edges_new, weights_new, list(unique_v)


def sort_vertex(label_list, vertex):
    """
    return sorted vertex according to the label_list
    :param label_list: the total sorted genre list
    :param vertex: the selected vertex
    :return: a list of the sorted vertex according to the label_list
    """
    sorted_vertex = []
    for genre in label_list:
        if genre in vertex:
            sorted_vertex.append(genre)
    return sorted_vertex


def get_tree(list_name, label_list, threshold, steps, similarity="default"):
    """
    use limited steps to implement the maximum spanning tree
    :param list_name: the 2D list genres
    :param label_list: the unique genre list sorted by f
    :param threshold: the edge weight bigger than the threshold will be kept
    :param steps: number of steps to implement the algorithm
    :param similarity: the similarity method
    :return: plot the tree and return the sorted nodes in the tree
    """
    co_score = get_co_score(list_name, similarity)
    max_spanning = MaxSpanningGraph(co_score.shape[0])
    max_spanning.graph = co_score
    (edges_old, weights_old), mst_set = max_spanning.prim_mst(steps, label_list)
    edges, edge_weights, v = edges_selected(threshold, edges_old, weights_old)
    # plot the graph
    tree_graph = ig.Graph()
    tree_graph.add_vertices(v)
    tree_graph.add_edges(edges)
    tree_graph.vs["label"] = tree_graph.vs["name"]
    tree_graph.es["label"] = edge_weights
    return sort_vertex(label_list, v), tree_graph, edges


