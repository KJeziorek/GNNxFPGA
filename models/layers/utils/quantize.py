import torch

def quantize_tensor(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    num_bits: int = 8,
                    signed: bool = False):
        
        '''Quantize tensor'''
        qmin = -2.**(num_bits-1) if signed else 0.
        qmax = 2.**(num_bits-1) - 1 if signed else 2.**num_bits - 1

        tensor_quant = zero_point + (tensor / scale).round_()
        
        tensor_quant = torch.clamp(tensor_quant, qmin, qmax)
        return tensor_quant
    
def dequantize_tensor(tensor_quant: torch.Tensor,
                      scale: torch.Tensor,
                      zero_point: torch.Tensor):
    
    '''Dequantize tensor'''
    tensor = scale * (tensor_quant - zero_point)
    return tensor