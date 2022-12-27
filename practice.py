import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

import torch_geometric.transforms as T
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name)
data = dataset[0]

# convert into a networkx object
g = to_networkx(data = data, to_undirected=True)


## -- your code here -- ##
## calculate the number of nodes, edges and density of the graph g
## you can use networkX library

num_nodes = g.number_of_nodes()
num_edges = g.number_of_edges()

numerator = 2 * num_edges
denomenator = num_nodes * (num_nodes - 1)
density = numerator / denomenator

print('Number of nodes: %d, number of edges: %d, density: %g'%(num_nodes, num_edges, density))

import matplotlib.pyplot as plt
def plot_degree_hist(G):
  freq = nx.degree_histogram(G) 

  plt.figure(figsize=(12,8))
  
  ## -- your code here -- ##
  ## It should be a log-log scale plot
  ## (x-axis: node degree, y-axis: degree frequency)
  degrees = range(len(freq))    
  plt.loglog(degrees, freq)
  # import numpy as np 
  # freq_log = np.log10(freq)
  # degrees_log = np.log10(degrees)
  # plt.plot(degrees_log, freq_log)
  
  plt.xlabel('Degree')
  plt.ylabel('Frequency')
  plt.show()

plot_degree_hist(g)

def compute_diameter(G):
  
  ## -- your code here -- ##
  ## If a graph is connected, you just use nx.diameter(.)
  ## If a graph is disconnected, you first extract the largest connected component as subgraph and compute the diameter of it
  ## 
  ## NetworkX functions will be useful such as nx.is_connected(.), nx.connected_components(.), nx.diameter(.)
  ## See https://networkx.org/documentation/stable/reference/algorithms/index.html
  ##
  ## For a large graph, approximations can be helpful (nx.approximation.diameter(.))
  ## See https://networkx.org/documentation/stable/reference/algorithms/approximation.html
  # print(nx.is_connected(g))  # false
  components_generator = nx.connected_components(g)
  component_diameters = []
  
  for components in components_generator:
     graph = g.subgraph(components).copy()
     component_diameters.append(nx.approximation.diameter(graph))
  import numpy as np
  diameter = np.max(component_diameters)
  return diameter

print(compute_diameter(g))