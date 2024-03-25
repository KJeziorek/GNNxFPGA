import numpy as np
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
        self.register_buffer('m', torch.tensor([-1], requires_grad=False))
        
        '''Initialize quantized version of scales.'''
        self.register_buffer('qscale_in', torch.tensor([-1], requires_grad=False))
        self.register_buffer('qscale_w', torch.tensor([-1], requires_grad=False))
        self.register_buffer('qscale_out', torch.tensor([-1], requires_grad=False))
        self.register_buffer('qscale_m', torch.tensor([-1], requires_grad=False))

        '''Initialize numbers of bits for model quantization and scales.'''
        self.register_buffer('num_bits_model', torch.tensor([num_bits], requires_grad=False))
        self.register_buffer('num_bits_scale', torch.tensor([-1], requires_grad=False))

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
            y = F.linear(msg, self.linear.weight, self.linear.bias)
            _ = self.norm(y)
            mean = Variable(self.norm.running_mean)
            var = Variable(self.norm.running_var)

        else:
            mean = Variable(self.norm.running_mean)
            var = Variable(self.norm.running_var)

        std = torch.sqrt(var + self.norm.eps)
        weight, bias = self.merge_norm(mean, std)

        '''Update weight observer and propagate message through linear layer.'''
        self.observer_w.update(weight)

        msg = F.linear(msg, FakeQuantize.apply(weight, self.observer_w), bias) # Merge batch normalization will always have bias

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
               num_bits: int = 32):
    
        '''Freeze model - quantize weights/bias and calculate scales'''
        if observer_in is not None:
            self.observer_in = observer_in
        if observer_out is not None:
            self.observer_out = observer_out

        self.num_bits_scale = torch.tensor([num_bits], requires_grad=False)

        scale_in = (2**num_bits-1) * self.observer_in.scale
        self.qscale_in = scale_in.round()
        self.observer_in.scale = scale_in.round() / (2**num_bits-1)

        scale_w = (2**num_bits-1) * self.observer_w.scale
        self.qscale_w = scale_w.round()
        self.observer_w.scale = scale_w.round() / (2**num_bits-1)

        scale_out = (2**num_bits-1) * self.observer_out.scale
        self.qscale_out = scale_out.round()
        self.observer_out.scale = scale_out.round() / (2**num_bits-1)

        m = (self.observer_w.scale * self.observer_in.scale / self.observer_out.scale)
        m = (2**num_bits-1) * m
        self.qscale_m = m.round()
        self.m = m.round() / (2**num_bits-1)

        std = torch.sqrt(self.norm.running_var + self.norm.eps)
        weight, bias = self.merge_norm(self.norm.running_mean, std)
        
        # self.linear = nn.Linear(self.input_dim + 3, self.output_dim, bias=True).eval()

        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(self.observer_w.quantize_tensor(weight))
            self.linear.weight = torch.nn.Parameter(self.linear.weight - self.observer_w.zero_point)

            self.linear.bias = torch.nn.Parameter(quantize_tensor(bias, scale=self.observer_in.scale*self.observer_w.scale,
                                            zero_point=0,
                                            num_bits=32,
                                            signed=True))

    def q_forward(self, 
                  node: torch.Tensor, 
                  features: torch.Tensor, 
                  edges: torch.Tensor,
                  first_layer: bool = False,
                  after_pool: bool = False):
        
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
        msg = (msg * self.m + self.observer_out.zero_point).floor() 
        msg = torch.clamp(msg, 0, 2**self.num_bits - 1)

        '''Update graph features.'''
        unique_positions, indices = torch.unique(edges[:,0], dim=0, return_inverse=True)
        expanded_indices = indices.unsqueeze(1).expand(-1, self.output_dim)
        pooled_features = torch.zeros((unique_positions.size(0), self.output_dim), dtype=features.dtype, device=features.device)
        pooled_features = pooled_features.scatter_reduce(0, expanded_indices, msg, reduce="amax", include_self=False) # Find max features for each node
        
        return pooled_features
    
    def get_parameters(self,
                       file_name: str = None):
        
        with open(file_name, 'w') as f:
            '''Save scales and zero points to file.'''
            f.write(f"Input scale ({int(self.num_bits_scale)} bit):\n {int(self.qscale_in)}\n")
            f.write(f"Input zero point:\n {int(self.observer_in.zero_point)}\n")
            f.write(f"Weight scale ({int(self.num_bits_scale)} bit):\n {int(self.qscale_w)}\n")
            f.write(f"Weight zero point:\n {int(self.observer_w.zero_point)}\n")
            f.write(f"Output scale ({int(self.num_bits_scale)} bit):\n {int(self.qscale_out)}\n")
            f.write(f"Output zero point:\n {int(self.observer_out.zero_point)}\n")
            f.write(f"M Scales ({int(self.num_bits_scale)} bit):\n {int(self.qscale_m)}\n")

            '''Save weights and bias to file.'''
            bias = torch.flip(self.linear.bias, [0])
            bias = bias.detach().cpu().numpy().astype(np.int32).tolist()
            weight = torch.flip(self.linear.weight, [1])
            weight = weight.detach().cpu().numpy().astype(np.int32).tolist()
            
            f.write(f"Weight ({int(self.num_bits_model)} bit):\n")
            for idx, w in enumerate(weight):
                f.write(f"weights_conv[{idx}] = {str(w).replace('[', '{').replace(']', '}') + ';'}\n")

            f.write(f"\nBias ({int(self.num_bits_model)} bit):\n")
            f.write(f"bias_conv = {str(bias).replace('[', '{').replace(']', '}') + ';'}\n")

            '''Save LUT for POS quantization to file.'''
            input_range = list(range(int(self.observer_in.min), int(self.observer_in.max + 1)))
            output_range = self.observer_in.quantize_tensor(torch.tensor(input_range).to(self.linear.weight.device)) - self.observer_in.zero_point
            output_range = output_range.detach().cpu().numpy().astype(np.int32).tolist()

            f.write(f"Input range ({int(self.num_bits_model)} bit):\n {input_range}\n")
            f.write(f"Output range ({int(self.num_bits_model)} bit):\n {output_range}\n")
        
        with open(file_name.replace('.txt', '.mem'), 'w') as f:
            for idx, we in enumerate(weight):
                bin_vec = [np.binary_repr(w+self.observer_w.zero_point.to(torch.int32).item(), width=9)[1:] for w in we]
                # Concat to bin_vec binary repr of bias
                bin_vec = bin_vec + [np.binary_repr(bias[len(bias)-idx-1], width=32)]
                dlugi_ciag_bitow = ''.join(bin_vec)
                wartosc_hex = hex(int(dlugi_ciag_bitow, 2))
                f.write(f"{str(wartosc_hex)[2:]}\n")

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias}, num_bits={self.num_bits})"
