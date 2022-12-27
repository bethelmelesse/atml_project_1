import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator
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

'''Our Model'''

class GNN(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers, dp_rate):
        
        super(GNN, self).__init__()
        
        # --- Your code here --- #
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                GCNConv(in_channels, out_channels)
            ]
            in_channels = c_hidden
        layers += [GCNConv(in_channels, c_out)]
        self.layers = torch.nn.Sequential(*layers)
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = dp_rate


    def forward(self, x, adj_t):
        
        # --- Your code here --- #
        for layer in self.layers:
            out = layer(x, adj_t)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x = out 
            
        out = F.log_softmax(x, dim=1)
        return out
    

    '''Train a GCN model'''
def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    train_x = data.x
    adj_t = data.adj_t
    out = model(train_x, adj_t)
    
    train_y = data.y[train_idx]
    out = out[train_idx]
    train_y = torch.squeeze(train_y, dim=1)
    loss = loss_fn(out, train_y)       
    
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc




import copy 
args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 100,
}

model = GNN(data.num_features, args['hidden_dim'],
            dataset.num_classes, args['num_layers'],
            args['dropout']).to(device)
# print(model)

# example = torch.rand(size=(1,data.num_features)).to(device)
# print(model(example, data.adj_t))
# print()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = F.nll_loss
evaluator = Evaluator(name='ogbn-arxiv')

best_model = None
best_valid_acc = 0

for epoch in range(1, 1 + args["epochs"]):
  loss = train(model, data, train_idx, optimizer, loss_fn)
  
  result = test(model, data, split_idx, evaluator)
  train_acc, valid_acc, test_acc = result
  if valid_acc > best_valid_acc:
    best_valid_acc = valid_acc
    best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')

best_result = test(best_model, data, split_idx, evaluator)
train_acc, valid_acc, test_acc = best_result
print(f'Best model: '
      f'Train: {100 * train_acc:.2f}%, '
      f'Valid: {100 * valid_acc:.2f}% '
      f'Test: {100 * test_acc:.2f}%')

print()