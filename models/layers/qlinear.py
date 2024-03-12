import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.utils.observer import Observer, FakeQuantize
from models.layers.utils.quantize import quantize_tensor, dequantize_tensor

class QuantLinear(nn.Module):
    '''Quantized version of Linear layer.'''
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

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.num_bits = num_bits

        '''Initialize quantization observers for input, weight and output tensors.'''
        self.observer_in = Observer(num_bits=num_bits)
        self.observer_w = Observer(num_bits=num_bits)
        self.observer_out = Observer(num_bits=num_bits)

        self.register_buffer('scales', torch.tensor([], requires_grad=False))

    def forward(self, 
                features: torch.Tensor):

        '''Standard forward pass of Linear layer.'''
        return self.linear(features)
    
    def calibration(self, 
                    features: torch.Tensor):
        
        '''Calibration forward for updating observers.'''

        '''Update input observer.'''
        self.observer_in.update(features)
        features = FakeQuantize.apply(features, self.observer_in)

        '''Update weight observer and propagate message through linear layer.'''
        self.observer_w.update(self.linear.weight.data)

        if self.bias:
            features = F.linear(features, FakeQuantize.apply(self.linear.weight, self.observer_w), self.linear.bias)
        else:
            features = F.linear(features, FakeQuantize.apply(self.linear.weight, self.observer_w))
        
        '''Update output observer and calculate output.'''
        self.observer_out.update(features)
        features = FakeQuantize.apply(features, self.observer_out)
        return features


    def freeze(self,
               observer_in: Observer = None,
               observer_out: Observer = None):
        
        '''Freeze model - quantize weights/bias and calculate scales'''
        if observer_in is not None:
            self.observer_in = observer_in
        if observer_out is not None:
            self.observer_out = observer_out

        self.scales.data = (self.observer_in.scale * self.observer_w.scale / self.observer_out.scale).data

        self.linear.weight.data = self.observer_w.quantize_tensor(self.linear.weight.data)
        self.linear.weight.data = self.linear.weight.data - self.observer_w.zero_point

        if self.bias:
            self.linear.bias.data = quantize_tensor(self.linear.bias.data, 
                                                    scale=self.qi.scale * self.qw.scale,
                                                    zero_point=0, 
                                                    num_bits=32, 
                                                    signed=True)

    def q_forward(self, 
                  features: torch.Tensor, 
                  first_layer: bool = False):
        
        '''Quantized forward pass of Linear layer.'''
        
        '''Quantize input features'''
        if first_layer:
            '''We need to quantize features.'''
            features = self.observer_in.quantize_tensor(features)
            features = features - self.observer_in.zero_point
        else:
            '''For other layers, we do not need to quantize features'''
            features = features - self.observer_in.zero_point
        
        features = self.linear(features)
        features = (features * self.scales).round_()
        features = features + self.observer_out.zero_point
        features.clamp(0., 2.**self.num_bits - 1.).round_()
        return features


    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias}, num_bits={self.num_bits})"
