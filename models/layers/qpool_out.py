import torch
from torch.nn import Module

from models.layers.utils.observer import Observer, FakeQuantize

class QuantGraphPoolOut(Module):
    def __init__(self, 
                 pool_size: int = 4, 
                 max_dimension: int = 256,
                 num_bits:int = 8):
        
        super(QuantGraphPoolOut, self).__init__()
        self.pool_size = pool_size
        self.max_dimension = max_dimension
        self.grid_size = max_dimension // pool_size

        self.two_dim = False

        '''Initialize quantization observers for input, weight and output tensors.'''
        self.observer_in = Observer(num_bits=num_bits)
        self.num_bits = num_bits

        self.register_buffer('scales', torch.tensor([], requires_grad=False))
    def forward(self, 
                vertices: torch.Tensor, 
                features: torch.Tensor):
        # Redukcja wymiarowości przestrzeni wierzchołków
        normalized_vertices = torch.div(vertices, self.pool_size, rounding_mode='floor').to(torch.int64)

        if self.two_dim:
            unique_positions, indices = torch.unique(normalized_vertices[:,:2], dim=0, return_inverse=True)
        else:
            unique_positions, indices = torch.unique(normalized_vertices, dim=0, return_inverse=True)

        pooled_features = torch.zeros((unique_positions.size(0), features.size(1)), dtype=features.dtype, device=features.device)

        if self.two_dim:
            output_features = torch.zeros((self.grid_size ** 2, features.size(1)), dtype=features.dtype, device=features.device)
        else:
            output_features = torch.zeros((self.grid_size ** 3, features.size(1)), dtype=features.dtype, device=features.device)

        # TODO - ("sum", "prod", "mean", "amax", "amin")
        pooled_features = pooled_features.scatter_reduce(0, indices.unsqueeze(1).expand(-1, features.size(1)), features, reduce="amax", include_self=False) #TODO Change to True
        
        # Przeliczenie indeksów dla output_features
        if self.two_dim:
            indices_1d = unique_positions[:, 0] * self.grid_size + unique_positions[:, 1]
        else:
            indices_1d = unique_positions[:, 0] * self.grid_size ** 2 + unique_positions[:, 1] * self.grid_size + unique_positions[:, 2]
        
        output_features[indices_1d] = pooled_features
        output_features = output_features.flatten()
        return output_features
    
    def calibration(self, 
                    vertices: torch.Tensor, 
                    features: torch.Tensor):

        # Redukcja wymiarowości przestrzeni wierzchołków
        normalized_vertices = torch.div(vertices, self.pool_size, rounding_mode='floor').to(torch.int64)

        if self.two_dim:
            unique_positions, indices = torch.unique(normalized_vertices[:,:2], dim=0, return_inverse=True)
        else:
            unique_positions, indices = torch.unique(normalized_vertices, dim=0, return_inverse=True)

        pooled_features = torch.zeros((unique_positions.size(0), features.size(1)), dtype=features.dtype, device=features.device)

        if self.two_dim:
            output_features = torch.zeros((self.grid_size ** 2, features.size(1)), dtype=features.dtype, device=features.device)
        else:
            output_features = torch.zeros((self.grid_size ** 3, features.size(1)), dtype=features.dtype, device=features.device)

        # TODO - ("sum", "prod", "mean", "amax", "amin")
        pooled_features = pooled_features.scatter_reduce(0, indices.unsqueeze(1).expand(-1, features.size(1)), features, reduce="amax", include_self=False) #TODO Change to True
        
        # Przeliczenie indeksów dla output_features
        if self.two_dim:
            indices_1d = unique_positions[:, 0] * self.grid_size + unique_positions[:, 1]
        else:
            indices_1d = unique_positions[:, 0] * self.grid_size ** 2 + unique_positions[:, 1] * self.grid_size + unique_positions[:, 2]
        
        output_features[indices_1d] = pooled_features
        output_features = output_features.flatten()

        return output_features
    
    def freeze(self,
               observer_in: Observer = None,
               observer_out: Observer = None):
        
        '''Freeze model - quantize weights/bias and calculate scales'''
        if observer_in is not None:
            self.observer_in = observer_in

    def q_forward(self, 
                    vertices: torch.Tensor, 
                    features: torch.Tensor):
        
        # Redukcja wymiarowości przestrzeni wierzchołków
        normalized_vertices = torch.div(vertices, self.pool_size, rounding_mode='floor').to(torch.int64)

        if self.two_dim:
            unique_positions, indices = torch.unique(normalized_vertices[:,:2], dim=0, return_inverse=True)
        else:
            unique_positions, indices = torch.unique(normalized_vertices, dim=0, return_inverse=True)

        pooled_features = torch.zeros((unique_positions.size(0), features.size(1)), dtype=features.dtype, device=features.device)

        if self.two_dim:
            output_features = torch.zeros((self.grid_size ** 2, features.size(1)), dtype=features.dtype, device=features.device) + self.observer_in.zero_point
        else:
            output_features = torch.zeros((self.grid_size ** 3, features.size(1)), dtype=features.dtype, device=features.device) + self.observer_in.zero_point

        # TODO - ("sum", "prod", "mean", "amax", "amin")
        pooled_features = pooled_features.scatter_reduce(0, indices.unsqueeze(1).expand(-1, features.size(1)), features, reduce="amax", include_self=False)

        # Przeliczenie indeksów dla output_features
        if self.two_dim:
            indices_1d = unique_positions[:, 0] * self.grid_size + unique_positions[:, 1]
        else:
            indices_1d = unique_positions[:, 0] * self.grid_size ** 2 + unique_positions[:, 1] * self.grid_size + unique_positions[:, 2]
        
        output_features[indices_1d] = pooled_features
        output_features = output_features.flatten()
        return output_features
    
    def __repr__(self):
        return f"{self.__class__.__name__}(pool_size={self.pool_size}, max_dimension={self.max_dimension})"