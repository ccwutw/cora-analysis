import os
import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as sp
import torch

from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community


__name__ == '__main__'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)


dataset = Planetoid(root=data_dir, name='Cora')
data = dataset[0]

print(f'Cora number of nodes: {data.num_nodes}')
print(f'Cora number of edges: {data.num_edges}')

edge_index = data.edge_index.numpy()
edge_example = edge_index[:, np.where(edge_index[0] == 30)[0]]
node_example = np.unique(edge_example.flatten())

example_fig = plt.figure(figsize=(10, 6))
G = nx.Graph()
G.add_nodes_from(node_example)
G.add_edges_from(list(zip(edge_example[0], edge_example[1])))
nx.draw_networkx(G, with_labels=False)
plt.show()
