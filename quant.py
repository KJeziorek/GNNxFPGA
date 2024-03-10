import torch
import torch.nn.functional as F
from torch.autograd import Function
torch.manual_seed(0)


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)

    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = torch.tensor([qmin], dtype=torch.float32).to(min_val.device)
    elif zero_point > qmax:
        # zero_point = qmax
        zero_point = torch.tensor([qmax], dtype=torch.float32).to(max_val.device)
    
    zero_point.round_()

    return scale, zero_point

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
 
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    
    return q_x
 
def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x - zero_point)

class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    
class QParam(torch.nn.Module):

    def __init__(self, num_bits=8):
        super(QParam, self).__init__()
        self.num_bits = num_bits
        scale = torch.tensor([], requires_grad=False)
        zero_point = torch.tensor([], requires_grad=False)
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('scale', scale)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def update(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)
        
        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)
        
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits)
    
    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)
    

class QLinear(torch.nn.Module):
    def __init__(self, layer, num_bits=8):
        super(QLinear, self).__init__()
        self.num_bits = num_bits
        self.layer = layer
        self.qi = QParam(num_bits)
        self.qw = QParam(num_bits)
        self.qo = QParam(num_bits)
        self.register_buffer('M', torch.tensor([], requires_grad=False))

    def freeze(self, qi=None, qo=None):

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        self.layer.weight.data = self.qw.quantize_tensor(self.layer.weight.data)
        self.layer.weight.data = self.layer.weight.data - self.qw.zero_point
        # self.layer.bias.data = quantize_tensor(self.layer.bias.data, scale=self.qi.scale * self.qw.scale,
        #                                            zero_point=0, num_bits=32, signed=True)
    
    def forward(self, x):
        self.qi.update(x)
        x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.layer.weight.data)
        # x = F.linear(x, FakeQuantize.apply(self.layer.weight, self.qw), self.layer.bias)
        x = F.linear(x, FakeQuantize.apply(self.layer.weight, self.qw))

        self.qo.update(x)
        x = FakeQuantize.apply(x, self.qo)
        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.layer(x)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x

# Define input and three linear layers
    
lin1 = torch.nn.Linear(1, 8, bias=False).eval()
lin2 = torch.nn.Linear(8, 8, bias=False).eval()
lin3 = torch.nn.Linear(8, 16, bias=False).eval()

qlin1 = QLinear(lin1, 8).eval()
qlin2 = QLinear(lin2, 4).eval()
qlin3 = QLinear(lin3, 4).eval()
for i in range(100):

    input = torch.randn(10, 1)

    output1 = lin1(input)
    output2 = lin2(output1)
    output3 = lin3(output2)
    
    qoutput1 = qlin1(input)
    qoutput2 = qlin2(qoutput1)
    qoutput3 = qlin3(qoutput2)

print(output3)
print(qoutput3)

qlin1.freeze()
qlin2.freeze(qi=qlin1.qo)
qlin3.freeze(qi=qlin2.qo)

input = qlin1.qi.quantize_tensor(input)
qoutput1 = qlin1.quantize_inference(input)
qoutput2 = qlin2.quantize_inference(qoutput1)
qoutput3 = qlin3.quantize_inference(qoutput2)

dequant = qlin3.qo.dequantize_tensor(qoutput3)
# print(qoutput3)
print(dequant)
# print(qlin1.layer.weight.data)

# print(qlin1.qo.scale, qlin1.qo.zero_point)
# print(qlin2.qi.scale, qlin2.qi.zero_point)

# print(qlin2.qo.scale, qlin2.qo.zero_point)
# print(qlin3.qi.scale, qlin3.qi.zero_point)