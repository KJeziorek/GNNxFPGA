import torch
import torch.nn as nn

class Observer(nn.Module):
    def __init__(self, 
                 num_bits:int = 8,
                 signed:bool = False):
        super().__init__()

        self.num_bits = num_bits
        self.signed = signed

        '''Initialize parameters for quantization'''
        self.scale = torch.tensor([], requires_grad=False)
        self.zero_point = torch.tensor([], requires_grad=False)
        self.min = torch.tensor([], requires_grad=False)
        self.max = torch.tensor([], requires_grad=False)
        self.register_buffer('scale', self.scale)
        self.register_buffer('zero_point', self.zero_point)
        self.register_buffer('min', self.min)
        self.register_buffer('max', self.max)

    def update(self, 
               tensor: torch.Tensor):
        
        '''Update parameters for quantization'''
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)

        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        self.scale, self.zero_point = self.calcScaleZeroPoint()

    def quantize_tensor(self, 
                        tensor: torch.Tensor):
        
        '''Quantize tensor'''
        qmin = -2**(self.num_bits-1) if self.signed else 0
        qmax = 2**(self.num_bits-1) - 1 if self.signed else 2**self.num_bits - 1

        tensor_quant = torch.round((tensor / self.scale,) + self.zero_point)
        tensor_quant = torch.clamp(tensor_quant, qmin, qmax)
        return tensor_quant
    
    def dequantize_tensor(self, 
                          tensor_quant: torch.Tensor):
        
        '''Dequantize tensor'''
        tensor = self.scale * (tensor_quant - self.zero_point)
        return tensor

    def calcScaleZeroPoint(self):

        '''Calculate scale and zero point for quantization'''
        qmin = -2**(self.num_bits-1) if self.signed else 0
        qmax = 2**(self.num_bits-1) - 1 if self.signed else 2**self.num_bits - 1

        scale = (self.max - self.min) / (qmax - qmin)
        zero_point = qmax - self.max  / scale

        if zero_point < qmin:
            zero_point = torch.tensor([qmin], dtype=torch.float32).to(self.min.device)
        elif zero_point > qmax:
            zero_point = torch.tensor([qmax], dtype=torch.float32).to(self.max.device)
        
        zero_point.round_()
        return scale, zero_point