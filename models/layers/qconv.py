import torch
import torch.nn as nn

from utils.observer import Observer

class QuantGraphConv(nn.Module):
    '''Quantized version of GraphConv layer.'''
    def __init__(self, 
                 input_dim: int = 1, 
                 output_dim: int = 4,
                 bias:bool = False,
                 num_bits:int = 8):
        super().__init__()
        
        '''Initialize standard layers.'''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.linear = nn.Linear(input_dim + 3, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        self.num_bits = num_bits

        '''Initialize quantization observers for input, weight and output tensors.'''
        self.observer_in = Observer(num_bits=num_bits)
        self.observer_w = Observer(num_bits=num_bits)
        self.observer_out = Observer(num_bits=num_bits)

    def forward(self, 
                node: torch.Tensor, 
                features: torch.Tensor, 
                edges: torch.Tensor):

        '''Standard forward pass of GraphConv layer.'''

        '''Calculate message for PointNet layer.'''
        pos_i = node[edges[:, 0]]
        pos_j = node[edges[:, 1]]
        x_j = features[edges[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)

        '''Propagate message through linear layer.'''
        msg = self.linear(msg)
        msg = self.norm(msg)
        msg = self.relu(msg)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edges[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax") # Find max features for each node
        
        return pooled_features
    
    def calibration(self, 
                    node: torch.Tensor, 
                    features: torch.Tensor, 
                    edges: torch.Tensor):
        
        '''Calibration forward for updating observers.'''

        '''Calculate message for PointNet layer.'''
        pos_i = node[edges[:, 0]]
        pos_j = node[edges[:, 1]]
        x_j = features[edges[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)

        '''Update input observer.'''
        self.observer_in.update(msg)
        

        '''Propagate message through linear layer.'''
        msg = self.linear(msg)
        msg = self.norm(msg)
        msg = self.relu(msg)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edges[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax") # Find max features for each node
        
        return pooled_features

    def q_forward(self, node, features, edges):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias})"
