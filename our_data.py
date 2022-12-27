
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

print()

'''Preparing Data'''
dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name,
                                 transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

print(data)
# print()
# print(data.adj_t)
# print()
# print(data.x)
print()