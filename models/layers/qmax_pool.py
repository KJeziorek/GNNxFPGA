import torch
from torch.nn import Module

from models.layers.utils.observer import Observer, FakeQuantize

class QuantGraphPooling(Module):
    def __init__(self, 
                 pool_size=4, 
                 max_dimension=256, 
                 num_bits=8,
                 only_vertices=False, 
                 self_loop=True):
        
        super(QuantGraphPooling, self).__init__()
        self.pool_size = pool_size
        self.max_dimension = max_dimension
        self.grid_size = max_dimension // pool_size
        self.only_vertices = only_vertices
        self.self_loop = self_loop

        self.average_positions = False

        self.num_bits = num_bits
        '''Initialize quantization observers for input, weight and output tensors.'''
        self.observer_in = Observer(num_bits=num_bits)
        self.observer_out = Observer(num_bits=num_bits)

        self.register_buffer('scales', torch.tensor([], requires_grad=False))

        self.register_buffer('qscale_in', torch.tensor([], requires_grad=False))
        self.register_buffer('qscale_out', torch.tensor([], requires_grad=False))
        self.register_buffer('qscale_m', torch.tensor([], requires_grad=False))

    def forward(self, 
                vertices, 
                features, 
                edges):
        
        # Reduce dimension of vertices to find indices with the same pool cells
        normalized_vertices = torch.div(vertices, self.pool_size, rounding_mode='floor').to(torch.int64)

        # Change vertices to original dimensions - OPTIONAL
        # normalized_vertices = normalized_vertices * self.pool_size
        
        #Find indices of unique positions
        unique_positions, indices = torch.unique(normalized_vertices, dim=0, return_inverse=True)
        # unique_positions, indices = torch.unique(normalized_vertices[:,:2], dim=0, return_inverse=True)

        # Uśredniona agregacja pozycji wierzchołków
        if self.average_positions:
            averaged_positions = torch.zeros((unique_positions.size(0), 3), dtype=vertices.dtype, device=vertices.device)
            unique_positions = averaged_positions.scatter_reduce(0, indices.unsqueeze(1).expand(-1,3), vertices, reduce="mean", include_self=False)

        # Agregacja maksymalnych cech dla każdej unikalnej pozycji
        # TODO - ("sum", "prod", "mean", "amax", "amin")
        pooled_features = torch.zeros((unique_positions.size(0), features.size(1)), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, indices.unsqueeze(1).expand(-1, features.size(1)), features, reduce="amax", include_self=False) #TODO Change to True

        # For potential pruning graph at the beginning
        if self.only_vertices:
            return unique_positions, pooled_features
        
        # Remove self loops (for filter out the same positions duplicates)
        edge_index = indices[edges]
        mask = edge_index[:, 0] != edge_index[:, 1]
        edge_index = edge_index[mask, :]

        edge_index = torch.unique(edge_index, dim=0)
        
        if self.self_loop:
            # Add self loops (to keep only one self loop for each unique position)
            edge_index = torch.cat((edge_index, torch.arange(unique_positions.size(0), device=edge_index.device).unsqueeze(1).expand(-1, 2)), dim=0)
        return unique_positions, pooled_features, edge_index
    
    def calibration(self, 
                    vertices, 
                    features, 
                    edges,
                    use_obs: bool = False,
                    min_max_diff: torch.tensor = None):
        
        if use_obs:
            '''Update input observer.'''
            self.observer_in.update(features)
            features = FakeQuantize.apply(features, self.observer_in)

        
        # Reduce dimension of vertices to find indices with the same pool cells
        normalized_vertices = torch.div(vertices, self.pool_size, rounding_mode='floor').to(torch.int64)

        # Change vertices to original dimensions - OPTIONAL
        # normalized_vertices = normalized_vertices * self.pool_size
        
        #Find indices of unique positions
        unique_positions, indices = torch.unique(normalized_vertices, dim=0, return_inverse=True)
        # unique_positions, indices = torch.unique(normalized_vertices[:,:2], dim=0, return_inverse=True)

        # Uśredniona agregacja pozycji wierzchołków
        if self.average_positions:
            averaged_positions = torch.zeros((unique_positions.size(0), 3), dtype=vertices.dtype, device=vertices.device)
            unique_positions = averaged_positions.scatter_reduce(0, indices.unsqueeze(1).expand(-1,3), vertices, reduce="mean", include_self=False)

        # Agregacja maksymalnych cech dla każdej unikalnej pozycji
        # TODO - ("sum", "prod", "mean", "amax", "amin")
        pooled_features = torch.zeros((unique_positions.size(0), features.size(1)), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, indices.unsqueeze(1).expand(-1, features.size(1)), features, reduce="amax", include_self=False) #TODO Change to True

        self.observer_out.update(pooled_features)
        if min_max_diff is not None:
            self.observer_out.update(min_max_diff)# this is maximum ranges for pooled POS diffs
        pooled_features = FakeQuantize.apply(pooled_features, self.observer_out)

        # For potential pruning graph at the beginning
        if self.only_vertices:
            return unique_positions, pooled_features
        
        # Remove self loops (for filter out the same positions duplicates)
        edge_index = indices[edges]
        mask = edge_index[:, 0] != edge_index[:, 1]
        edge_index = edge_index[mask, :]

        edge_index = torch.unique(edge_index, dim=0)
        
        if self.self_loop:
            # Add self loops (to keep only one self loop for each unique position)
            edge_index = torch.cat((edge_index, torch.arange(unique_positions.size(0), device=edge_index.device).unsqueeze(1).expand(-1, 2)), dim=0)
        return unique_positions, pooled_features, edge_index
    
    def freeze(self,
               observer_in: Observer = None,
               observer_out: Observer = None,
               num_bits: int = 16):
        
        '''Freeze model - quantize weights/bias and calculate scales'''
        if observer_in is not None:
            self.observer_in = observer_in
        if observer_out is not None:
            self.observer_out = observer_out

        self.scales.data = self.observer_in.scale / self.observer_out.scale
        
    def q_forward(self, 
                  vertices, 
                  features, 
                  edges):
        
        # Reduce dimension of vertices to find indices with the same pool cells
        normalized_vertices = torch.div(vertices, self.pool_size, rounding_mode='floor').to(torch.int64)

        # Change vertices to original dimensions - OPTIONAL
        # normalized_vertices = normalized_vertices * self.pool_size
        
        #Find indices of unique positions
        unique_positions, indices = torch.unique(normalized_vertices, dim=0, return_inverse=True)
        # unique_positions, indices = torch.unique(normalized_vertices[:,:2], dim=0, return_inverse=True)

        # Uśredniona agregacja pozycji wierzchołków
        if self.average_positions:
            averaged_positions = torch.zeros((unique_positions.size(0), 3), dtype=vertices.dtype, device=vertices.device)
            unique_positions = averaged_positions.scatter_reduce(0, indices.unsqueeze(1).expand(-1,3), vertices, reduce="mean", include_self=False)

        # Agregacja maksymalnych cech dla każdej unikalnej pozycji
        # TODO - ("sum", "prod", "mean", "amax", "amin")
        pooled_features = torch.zeros((unique_positions.size(0), features.size(1)), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, indices.unsqueeze(1).expand(-1, features.size(1)), features, reduce="amax", include_self=False) #TODO Change to True

        pooled_features = pooled_features - self.observer_in.zero_point

        pooled_features = (pooled_features * self.scales).round() 
        pooled_features = pooled_features + self.observer_out.zero_point
        pooled_features = torch.clamp(pooled_features, 0, 2**self.num_bits - 1)

        # For potential pruning graph at the beginning
        if self.only_vertices:
            return unique_positions, pooled_features
        
        # Remove self loops (for filter out the same positions duplicates)
        edge_index = indices[edges]
        mask = edge_index[:, 0] != edge_index[:, 1]
        edge_index = edge_index[mask, :]

        edge_index = torch.unique(edge_index, dim=0)
        
        if self.self_loop:
            # Add self loops (to keep only one self loop for each unique position)
            edge_index = torch.cat((edge_index, torch.arange(unique_positions.size(0), device=edge_index.device).unsqueeze(1).expand(-1, 2)), dim=0)
        return unique_positions, pooled_features, edge_index
    
    def __repr__(self):
        return f"{self.__class__.__name__}(pool_size={self.pool_size}, max_dimension={self.max_dimension})"