import torch

def quantize_tensor(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    num_bits: int = 8,
                    signed: bool = False):
        
        '''Quantize tensor'''
        qmin = -2**(num_bits-1) if signed else 0
        qmax = 2**(num_bits-1) - 1 if signed else 2**num_bits - 1

        tensor_quant = tensor / scale + zero_point
        
        tensor_quant = torch.clamp(tensor_quant, qmin, qmax).round()
        tensor_quant = torch.round_((tensor / scale) + zero_point)
        return tensor_quant
    
def dequantize_tensor(tensor_quant: torch.Tensor,
                      scale: torch.Tensor,
                      zero_point: torch.Tensor):
    
    '''Dequantize tensor'''
    tensor = scale * (tensor_quant - zero_point)
    return tensor