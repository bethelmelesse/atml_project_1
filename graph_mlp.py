import torch

import torch_geometric.transforms as T
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator

print()
# transform into a sparse tensor
dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name,
                                 transform=T.ToSparseTensor())

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
print(data)

class MLP(torch.nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers, dp_rate):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super(MLP, self).__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                torch.nn.Linear(in_channels, out_channels)
            ]
            in_channels = c_hidden
        layers += [torch.nn.Linear(in_channels, c_out)]
        self.layers = torch.nn.Sequential(*layers)
        self.softmax = torch.nn.LogSoftmax()
        self.dropout = dp_rate

    def forward(self, x):
      ## --- your code here --- ##
      ## For each layer (use self.layers), add ReLU activation (torch.nn.functional.relu) and 
      ##   dropout (torch.nn.functional.dropout) when it is in training mode (you can use self.training)
      ## See https://pytorch.org/docs/stable/nn.functional.html
      ## return output ('out') after the softmax layer (use self.softmax)
      for layer in self.layers:
        out = layer(x)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        x = out 

      out = F.log_softmax(out, dim=1)
      return out


def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    ## fill in the parameters of each function
    # out = model(...)
    # loss = loss_fn(...)
    train_x = data.x[train_idx]
    out = model(train_x)
    
    train_y = data.y[train_idx] 
    train_y = torch.squeeze(train_y, dim=1)
    loss = loss_fn(out, train_y)         
    
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, split_idx, evaluator):
    model.eval()

    ## fill in the parameters 
    # out = model(...)

    out = model(data.x)
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

model = MLP(data.num_features, args['hidden_dim'],
            dataset.num_classes, args['num_layers'],
            args['dropout']).to(device)

# print(model)
# example = torch.rand(size=(1,data.num_features)).to(device)
# print(model(example))

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