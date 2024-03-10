import torch
from torch.nn import Module

class GraphConv(Module):
    def __init__(self, input_dim=1, output_dim=4, bias=False):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.linear = torch.nn.Linear(input_dim + 3, output_dim, bias=bias)
        self.norm = torch.nn.BatchNorm1d(output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, node, features, edges):
        pos_i = node[edges[:, 0]]
        pos_j = node[edges[:, 1]]
        x_j = features[edges[:, 1]]

        msg = torch.cat((x_j, pos_j - pos_i), dim=1)
        msg = self.linear(msg)
        msg = self.norm(msg)
        msg = self.relu(msg)

        unique_positions, indices = torch.unique(edges[:,0], dim=0, return_inverse=True)

        # Rozszerzenie indeksów do kształtu pasującego do msg
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)

        # Inicjalizacja tensora wynikowego
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=features.dtype, device=features.device)

        # Agregacja cech bez użycia pętli
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax") #TODO Change to True
        
        return pooled_features
    
    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias})"