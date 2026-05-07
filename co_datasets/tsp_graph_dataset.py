# update at 2024/12/13

import glob
import os
import pickle as pickle
import numpy as np
import torch
import networkx as nx  

from torch_geometric.data import Data as GraphData

class TSPGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_file,sparse_factor=-1):
        """
        Args:
            data_file (str): Path to the .gpickle files (glob pattern supported).
        """
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.file_lines = glob.glob(data_file)
        print(f'Loaded "{data_file}" with {len(self.file_lines)} examples')

    def __len__(self):
        return len(self.file_lines)

    def get_example(self, idx):
        # Translated English comment.
        with open(self.file_lines[idx], "rb") as f:
            graph = nx.read_gpickle(f)

        # Translated English comment.
        num_nodes = graph.number_of_nodes()

        # Translated English comment.
        edges = np.array(graph.edges, dtype=np.int64)
        edge_relations = np.array([graph[u][v].get('weight', 0.0) for u, v in graph.edges], dtype=np.float32)

        # Translated English comment.
        # Translated English comment.
        # Translated English comment.

        # Translated English comment.
        #edge_set = {tuple(edge) for edge in edges}

        # Translated English comment.
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         if (i, j) not in edge_set:
        #             if i == j:
        # Translated English comment.
        #                 edges = np.append(edges, [[i, j]], axis=0)
        #                 edge_relations = np.append(edge_relations, [1], axis=0)
        #             else:
        # Translated English comment.
        #                 edges = np.append(edges, [[i, j]], axis=0)
        #                 edge_relations = np.append(edge_relations, [0], axis=0)
        # Translated English comment.
        # adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # Translated English comment.
        # for edge, relation in zip(edges, edge_relations):
        #     u, v = edge
        #     adj_matrix[u, v] = relation
        

        # Translated English comment.
        node_labels = np.array([graph.nodes[node].get('label', 0) for node in range(num_nodes)], dtype=np.int64)

        #return adj_matrix, num_nodes, edges, edge_relations, node_labels
        return  num_nodes, edges, edge_relations, node_labels
    def __getitem__(self, idx):
        #adj_matrix,num_nodes, edge_index, edge_relations, node_labels = self.get_example(idx)
        num_nodes, edge_index, edge_relations, node_labels = self.get_example(idx)
        graph_data = GraphData(
            edge_index=torch.from_numpy(edge_index),  #.T? PyTorch Geometric expects edge_index to be [2, num_edges] 
            edge_attr=torch.from_numpy(edge_relations),  
            x=torch.from_numpy(node_labels).long()  
        )

        point_indicator = np.array([num_nodes], dtype=np.int64)
        return (
            torch.LongTensor(np.array([idx], dtype=np.int64)),
            graph_data,
            torch.from_numpy(point_indicator).long(),
            num_nodes,
            #torch.from_numpy(adj_matrix).float(),
        )
        
