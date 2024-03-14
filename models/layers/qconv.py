import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.layers.utils.observer import Observer, FakeQuantize
from models.layers.utils.quantize import quantize_tensor, dequantize_tensor

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

        self.num_bits = num_bits

        '''Initialize quantization observers for input, weight and output tensors.'''
        self.observer_in = Observer(num_bits=num_bits)
        self.observer_w = Observer(num_bits=num_bits)
        self.observer_out = Observer(num_bits=num_bits)

        self.register_buffer('scales', torch.tensor([], requires_grad=False))

        self.register_buffer('qscale_in', torch.tensor([], requires_grad=False))
        self.register_buffer('qscale_w', torch.tensor([], requires_grad=False))
        self.register_buffer('qscale_out', torch.tensor([], requires_grad=False))
        self.register_buffer('qscale_m', torch.tensor([], requires_grad=False))


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
        # msg = self.norm(msg)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edges[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax", include_self=False)
        return pooled_features
    
    def merge_norm(self,
                   mean: torch.Tensor,
                   std: torch.Tensor):
        '''Merge batch normalization with Linear layer.'''

        if self.norm.affine:
            gamma = self.norm.weight
            weight = (gamma / std).unsqueeze(1) * self.linear.weight
            if self.bias:
                bias = self.linear.bias * gamma - mean * gamma + self.norm.bias
            else:
                bias = self.norm.bias - mean * (gamma / std)
        else:
            gamma = 1 / std
            weight = self.linear.weight * gamma.unsqueeze(1)
            if self.bias:
                bias = self.linear.bias * gamma - mean * gamma
            else:
                bias = - mean * gamma
        
        return weight, bias
    
    def calibration(self, 
                    node: torch.Tensor, 
                    features: torch.Tensor, 
                    edges: torch.Tensor,
                    use_obs: bool = False):
        
        '''Calibration forward for updating observers.'''

        '''Calculate message for PointNet layer.'''
        pos_i = node[edges[:, 0]]
        pos_j = node[edges[:, 1]]
        x_j = features[edges[:, 1]]
        msg = torch.cat((x_j, pos_j - pos_i), dim=1)

        if use_obs:
            '''Update input observer.'''
            self.observer_in.update(msg)
            msg = FakeQuantize.apply(msg, self.observer_in)

        '''Update batch normalization observer.'''
        if self.training:
            _ = self.norm(msg)
            pass #TODO - add implementation for QAT, for now use only with .eval()
        else:
            mean = Variable(self.norm.running_mean)
            var = Variable(self.norm.running_var)

        std = torch.sqrt(var + self.norm.eps)
        weight, bias = self.merge_norm(mean, std)

        '''Update weight observer and propagate message through linear layer.'''
        # self.observer_w.update(weight.data)
        self.observer_w.update(self.linear.weight)
        # TODO - Merge batch normalization will always have bias
        # msg = F.linear(msg, FakeQuantize.apply(weight, self.observer_w), bias)

        msg = F.linear(msg, FakeQuantize.apply(self.linear.weight, self.observer_w))
        
        '''Update output observer and calculate output.'''
        '''We calibrate based on the output of the Linear and also for diff POS for next layer'''
        self.observer_out.update(msg)
        self.observer_out.update(pos_j-pos_i)
        msg = FakeQuantize.apply(msg, self.observer_out)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edges[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax", include_self=False)
        
        return pooled_features


    def freeze(self,
               observer_in: Observer = None,
               observer_out: Observer = None,
               num_bits: int = 16):
        
        '''Freeze model - quantize weights/bias and calculate scales'''
        if observer_in is not None:
            self.observer_in = observer_in
        if observer_out is not None:
            self.observer_out = observer_out

        scale_in = (2**num_bits-1) * self.observer_in.scale.data
        self.observer_in.scale.data = scale_in.floor() / (2**num_bits-1)
        self.qscale_in.data = scale_in

        scale_w = (2**num_bits-1) * self.observer_w.scale.data
        self.observer_w.scale.data = scale_w.floor() / (2**num_bits-1)
        self.qscale_w.data = scale_in

        scale_out = (2**num_bits-1) * self.observer_out.scale.data
        self.observer_out.scale.data = scale_out.floor() / (2**num_bits-1)
        self.qscale_out.data = scale_in

        scale_m = (self.observer_w.scale * self.observer_in.scale / self.observer_out.scale).data
        scale_m = (2**num_bits-1) * scale_m
        self.scales.data = scale_m.floor() / (2**num_bits-1)
        self.qscale_m.data = scale_m.floor() / (2**num_bits-1)

        std = torch.sqrt(self.norm.running_var + self.norm.eps)
        # weight, bias = self.merge_norm(self.norm.running_mean, std)
        
        self.linear = nn.Linear(self.input_dim + 3, self.output_dim)

        # self.linear.weight.data = self.observer_w.quantize_tensor(weight)
        self.linear.weight.data = self.observer_w.quantize_tensor(self.linear.weight)
        self.linear.weight.data = self.linear.weight.data - self.observer_w.zero_point

        # swap first 
        # weight = torch.tensor([[176.,  18., 176.,  98., 168., 255., 222., 190.],
        # [ 58.,  27.,  75.,  75.,  93., 162.,  65., 240.],
        # [129., 224., 186.,  43., 213., 150., 168.,   0.],
        # [ 70.,  10.,  40.,  10.,  22., 221.,  64., 233.]]).T - self.observer_w.zero_point
        # self.linear.weight.data = torch.flip(weight, [0])
        # self.linear.bias.data = quantize_tensor(bias, scale=self.observer_in.scale*self.observer_w.scale,
        #                                    zero_point=0,
        #                                    num_bits=32,
        #                                    signed=True)


    def q_forward(self, 
                  node: torch.Tensor, 
                  features: torch.Tensor, 
                  edges: torch.Tensor,
                  first_layer: bool = False):
        
        '''Quantized forward pass of GraphConv layer.'''

        '''Quantize input features'''
        if first_layer:
            '''We need to quantize both features and POS for the first layer.'''
            pos_i = node[edges[:, 0]]
            pos_j = node[edges[:, 1]]
            x_j = features[edges[:, 1]]
            msg = torch.cat((x_j, pos_j - pos_i), dim=1)
            msg = self.observer_in.quantize_tensor(msg)
        else:
            '''For other layers, we only quantize POS, because features are already quantized.'''
            pos_i = node[edges[:, 0]]
            pos_j = node[edges[:, 1]]

            pos = self.observer_in.quantize_tensor(pos_j - pos_i)
            msg = torch.cat((features[edges[:, 1]], pos), dim=1)

        msg = msg - self.observer_in.zero_point
        msg = self.linear(msg)
        msg = (msg*self.scales).floor() + self.observer_out.zero_point
        msg = torch.clamp_(msg, 0, 2**self.num_bits - 1)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edges[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax", include_self=False) # Find max features for each node
        
        return pooled_features
    
    def get_parameters(self):
        print("Input scale:", self.qscale_in)
        print("Input zero point", self.observer_in.zero_point)
        print("Weight scale:", self.qscale_in)
        print("Weight zero point", self.observer_w.zero_point)
        print("Output scale:", self.qscale_in)
        print("Output zero point", self.observer_out.zero_point)

        print("Weight", self.linear.weight)

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias}, num_bits={self.num_bits})"
