import os
import os
import random
import re
from collections import defaultdict
from itertools import *

import numpy as np
import scipy
import scipy.sparse as sp
import torch
from math import e
import math

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class DataGenerator(object):
    def __init__(self, args):
        self.in_f_d = args.in_f_d
        self.batch_n = args.batch_n  # batch number
        self.sample_g_n = args.sample_g_n  # train sample graph number
        self.test_sample_g_n = args.test_sample_g_n  # test sample graph number
        self.A_n = args.A_n
        self.graphpath = args.graphpath
        self.class_n = 4
        self.batch_total_n = 20
        self.batch_train_n = args.batch_train_n
        # self.batch_evaluate_n = 5
        self.k_hop = 3
        self.weight_metric = 1  # 1: k-hop common neighbor num, 2: jacard index, 3: adar index, 4: pagerank

        a_text_f = np.zeros((self.A_n, args.in_f_d))
        a_t_e_f = open(args.datapath + "feature.txt", "r")
        for line in islice(a_t_e_f, 0, None):
            values = line.split()
            index = int(values[0])
            embeds = np.asarray(values[1:], dtype='float32')
            a_text_f[index] = embeds
        a_t_e_f.close()

        self.a_text_f = a_text_f
    # sys.exit()

    def next_batch(self):
        batch_adj = []
        batch_unweight_adj = []
        batch_features = []
        batch_labels = []
        batch_idx_train = []
        batch_idx_evaluate = []
        batch_weight_matrix_train = []

        for i in range(self.batch_n):
            g_id = np.random.randint(self.sample_g_n)
            adj, unweight_adj, features, labels, idx_train, idx_test, weight_matrix = self.load_data(
                g_id, 1)

            batch_adj.append(adj)
            batch_unweight_adj.append(unweight_adj)
            batch_features.append(features)
            batch_labels.append(labels)
            batch_idx_train.append(idx_train)
            batch_idx_evaluate.append(idx_test)
            batch_weight_matrix_train.append(weight_matrix)

        return batch_adj, batch_unweight_adj, batch_features, batch_labels, batch_idx_train, batch_idx_evaluate, batch_weight_matrix_train

    def test_batch(self):
        batch_adj = []
        batch_unweight_adj = []
        batch_features = []
        batch_labels = []
        batch_idx_train = []
        batch_idx_evaluate = []
        batch_weight_matrix_train = []

        # total_count = 0.0
        for i in range(self.test_sample_g_n):
            # g_id = np.random.randint(self.sample_g_n)
            adj, unweight_adj, features, labels, idx_train, idx_test, weight_matrix = self.load_data(i, 0)
            batch_adj.append(adj)
            batch_unweight_adj.append(unweight_adj)
            batch_features.append(features)
            batch_labels.append(labels)
            batch_idx_train.append(idx_train)
            batch_idx_evaluate.append(idx_test)
            batch_weight_matrix_train.append(weight_matrix)

        # 	total_count += len(labels)

        # print total_count / self.test_sample_g_n

        return batch_adj, batch_unweight_adj, batch_features, batch_labels, batch_idx_train, batch_idx_evaluate, batch_weight_matrix_train

    def load_data(self, g_id, train_test):
        if train_test == 1:
            id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id) + "_label"),
                                     dtype=np.dtype(int))
        elif train_test == 0:
            id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "test_graph_" + str(g_id) + "_label"),
                                     dtype=np.dtype(int))

        labels = encode_onehot(id_label[:, -1])

        idx = np.array(id_label[:, 0], dtype=np.int32)

        # text feature
        features = np.zeros((len(idx), self.in_f_d))
        for i in range(len(idx)):
            features[i] = self.a_text_f[idx[i]]
        features = sp.csr_matrix(features, dtype=np.float32)
        features = normalize_features(features)

        idx_map = {j: i for i, j in enumerate(idx)}

        if train_test == 1:
            edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id)), dtype=np.int32)
        elif train_test == 0:
            edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "test_graph_" + str(g_id)),
                                            dtype=np.int32)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(id_label.shape[0], id_label.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # print (edges[0])
        # unweighted
        adj = (adj + sp.eye(adj.shape[0])) > 0
        adj = adj.astype(int)
        unweight_adj = adj
        adj = normalize_adj(adj)

        adj = torch.FloatTensor(np.array(adj.todense()))
        unweigh_adj = torch.LongTensor(np.array(unweight_adj.todense()))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        # labels = torch.FloatTensor(labels)
        # idx_train = torch.LongTensor(idx_train)

        graph_adj = defaultdict(list)
        for i in range(len(edges)):
            graph_adj[edges[i][0]].append(edges[i][1])
            graph_adj[edges[i][1]].append(edges[i][0])

        label_id_list = [[] for k in range(self.class_n)]
        for i in range(len(labels)):
            label_id_list[int(labels[i])].append(i)

        if train_test == 1:
            idx_all = [[] for j in range(self.class_n)]
            for i in range(self.class_n):
                idx_all[i] = label_id_list[i]

            idx_train = [[0 for i in range(self.batch_train_n)] for j in range(self.class_n)]
            batch_test_n = self.batch_total_n - self.batch_train_n

            idx_test = []
            # idx_test = [[0 for i in range(batch_test_n)] for j in range(self.class_n)]
            for i in range(self.class_n):
                for j in range(self.batch_train_n):
                    idx_train[i][j] = idx_all[i][j]
                for k in range(self.batch_train_n, len(idx_all[i])):
                    # idx_test[i][k - self.batch_train_n] = idx_all[i][k]
                    idx_test.append(idx_all[i][k])

            weight_matrix_all = []
            graph_adj_sparse = sparse_matrix_transform(graph_adj)
            if self.weight_metric == 4:
                matrix_inv = pagerank(graph_adj_sparse, len(adj))

            for l in range(self.class_n):
                weight_matrix = [[0 for i in range(self.batch_train_n)] for j in range(self.batch_train_n)]
                for i in range(self.batch_train_n):
                    weight_matrix[i][i] = 1.0
                    for j in range(i + 1, self.batch_train_n):
                        src_id = idx_train[l][i]
                        end_id = idx_train[l][j]
                        if self.weight_metric == 1:
                            weight_temp = k_hop_common_neigh(graph_adj, src_id, end_id)
                        elif self.weight_metric == 2:
                            weight_temp = jaccard_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 3:
                            weight_temp = adar_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 4:
                            weight_temp = matrix_inv[src_id][end_id]
                        weight_matrix[i][j] = weight_temp
                        weight_matrix[j][i] = weight_temp
                weight_matrix_all.append(weight_matrix)

            weight_matrix_all = torch.FloatTensor(weight_matrix_all)

            # print weight_matrix_all[0]

            return adj, unweigh_adj, features, labels, idx_train, idx_test, weight_matrix_all

        elif train_test == 0:
            idx_all = [[] for j in range(self.class_n)]
            for i in range(self.class_n):
                idx_all[i] = label_id_list[i]

            idx_train = [[0 for i in range(self.batch_train_n)] for j in range(self.class_n)]
            batch_test_n = self.batch_total_n - self.batch_train_n

            idx_test = []
            # idx_test = [[0 for i in range(batch_test_n)] for j in range(self.class_n)]
            for i in range(self.class_n):
                for j in range(self.batch_train_n):
                    idx_train[i][j] = idx_all[i][j]
                for k in range(self.batch_train_n, len(idx_all[i])):
                    # idx_test[i][k - self.batch_train_n] = idx_all[i][k]
                    idx_test.append(idx_all[i][k])

            weight_matrix_all = []
            graph_adj_sparse = sparse_matrix_transform(graph_adj)
            if self.weight_metric == 4:
                matrix_inv = pagerank(graph_adj_sparse, len(adj))
            for l in range(self.class_n):
                weight_matrix = [[0 for i in range(self.batch_train_n)] for j in range(self.batch_train_n)]
                for i in range(self.batch_train_n):
                    weight_matrix[i][i] = 1.0
                    for j in range(i + 1, self.batch_train_n):
                        src_id = idx_train[l][i]
                        end_id = idx_train[l][j]
                        if self.weight_metric == 1:
                            weight_temp = k_hop_common_neigh(graph_adj, src_id, end_id)
                        elif self.weight_metric == 2:
                            weight_temp = jaccard_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 3:
                            weight_temp = adar_index(graph_adj, src_id, end_id)
                        elif self.weight_metric == 4:
                            weight_temp = matrix_inv[src_id][end_id]
                        # weight_temp = math.exp(- weight_temp)
                        weight_matrix[i][j] = weight_temp
                        weight_matrix[j][i] = weight_temp
                weight_matrix_all.append(weight_matrix)

            weight_matrix_all = torch.FloatTensor(weight_matrix_all)

            # print weight_matrix_all[0]

            return adj, unweigh_adj, features, labels, idx_train, idx_test, weight_matrix_all

    def test_graph_analysis(self):
        total_count = 0.0
        for g_id in range(100):
            id_label = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id) + "_label"),
                                     dtype=np.dtype(int))
            edges_unordered = np.genfromtxt("{}{}.txt".format(self.graphpath, "graph_" + str(g_id)), dtype=np.int32)
            idx = np.array(id_label[:, 0], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(idx)}

            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
                edges_unordered.shape)

            total_count += len(edges)
        print(total_count)


def bfs(graph, start, end):
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # print path
        # get the last node from the path
        node = path[-1]
        # print node
        # path found
        if node == end:
            return path
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)


def k_hop_common_neigh(graph, src_id, end_id):  # k = 3
    src_neigh = []
    end_neigh = []

    node = [src_id, end_id]
    for l in range(len(node)):
        node_id = node[l]
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            if l == 0:
                src_neigh.append(neigh_id)
            else:
                end_neigh.append(neigh_id)
            for j in range(len(graph[neigh_id])):
                neigh_id_2 = int(graph[neigh_id][j])
                if l == 0:
                    src_neigh.append(neigh_id_2)
                else:
                    end_neigh.append(neigh_id_2)
                for k in range(len(graph[neigh_id_2])):
                    neigh_id_3 = int(graph[neigh_id_2][k])
                    if l == 0:
                        src_neigh.append(neigh_id_3)
                    else:
                        end_neigh.append(neigh_id_3)

    intersect = list(set(src_neigh) & set(end_neigh))
    # union = list(set(src_neigh) | set(end_neigh))

    weight = 1 / (1 + math.exp(- len(intersect)))

    return weight


def jaccard_index(graph, src_id, end_id):
    src_neigh = []
    end_neigh = []
    node = [src_id, end_id]
    for l in range(len(node)):
        node_id = node[l]
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            if l == 0:
                src_neigh.append(neigh_id)
            else:
                end_neigh.append(neigh_id)

    intersect_list = list(set(src_neigh) & set(end_neigh))
    union_list = list(set(src_neigh).union(end_neigh))

    weight = 0.5 + 0.5 * float(len(intersect_list)) / len(union_list)

    return weight


def adar_index(graph, src_id, end_id):
    src_neigh = []
    end_neigh = []
    node = [src_id, end_id]
    for l in range(len(node)):
        node_id = node[l]
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            if l == 0:
                src_neigh.append(neigh_id)
            else:
                end_neigh.append(neigh_id)

    intersect_list = list(set(src_neigh) & set(end_neigh))

    weight = 1.0 / (math.log(len(intersect_list) + 5))

    return weight


def matrix_transform(graph, A_n):
    graph_matrix = [[0 for i in range(A_n)] for j in range(A_n)]
    for node_id in range(len(graph)):
        for i in range(len(graph[node_id])):
            neigh_id = int(graph[node_id][i])
            graph_matrix[node_id][neigh_id] = 1
            graph_matrix[neigh_id][node_id] = 1

    return graph_matrix


def sparse_matrix_transform(graph):
    row_ind = [k for k, v in graph.items() for _ in range(len(v))]
    col_ind = [i for ids in graph.values() for i in ids]

    X = sp.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)))

    return X


def pagerank(graph, node_n):
    # graph_inv = np.linalg.pinv(graph, rcond=1e-10)
    identity_m = scipy.sparse.identity(node_n)
    new_graph = identity_m.todense() - 0.1 * graph.todense()
    # new_graph = scipy.sparse.csr_matrix(new_graph)
    graph_inv = np.linalg.pinv(new_graph)

    max_v = np.matrix.max(graph_inv)
    min_v = np.matrix.min(graph_inv)

    graph_inv = (graph_inv - min_v) / (max_v - min_v)
    graph_inv = graph_inv.tolist()

    # print (graph_inv[src][end])

    return graph_inv
