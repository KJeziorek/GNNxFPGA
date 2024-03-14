import torch
from torch.nn import Module, Dropout
from models.layers.qconv import QuantGraphConv
from models.layers.qpool_out import QuantGraphPoolOut
from models.layers.qlinear import QuantLinear
from models.layers.qrelu import QuantReLU
from models.layers.max_pool import GraphPooling
    
class Model(Module):
    def __init__(self, 
                 input_dimension: int = 256,
                 bias: bool = False, 
                 num_classes: int = 100, 
                 num_bits: int = 8):
        super(Model, self).__init__()

        self.conv1 = QuantGraphConv(input_dim=1, output_dim=8, bias=bias, num_bits=num_bits)
        self.relu1 = QuantReLU(num_bits=num_bits)
        self.conv2 = QuantGraphConv(input_dim=8, output_dim=16, bias=bias, num_bits=num_bits)
        self.relu2 = QuantReLU(num_bits=num_bits)

        self.max_pool1 = GraphPooling(pool_size=4, max_dimension=input_dimension, only_vertices=False, self_loop=True)

        self.conv3 = QuantGraphConv(input_dim=16, output_dim=32, bias=bias, num_bits=num_bits)
        self.relu3 = QuantReLU(num_bits=num_bits)
        self.conv4 = QuantGraphConv(input_dim=32, output_dim=32, bias=bias, num_bits=num_bits)
        self.relu4 = QuantReLU(num_bits=num_bits)

        self.max_pool2 = GraphPooling(pool_size=4, max_dimension=64)

        self.conv5 = QuantGraphConv(input_dim=32, output_dim=64, bias=bias, num_bits=num_bits)
        self.relu5 = QuantReLU(num_bits=num_bits)
        self.conv6 = QuantGraphConv(input_dim=64, output_dim=64, bias=bias, num_bits=num_bits)
        self.relu6 = QuantReLU(num_bits=num_bits)
        self.conv7 = QuantGraphConv(input_dim=64, output_dim=64, bias=bias, num_bits=num_bits)
        self.relu7 = QuantReLU(num_bits=num_bits)

        self.out = QuantGraphPoolOut(pool_size=4, max_dimension=16)
        self.dropout = Dropout(p=0.5)
        self.linear = QuantLinear(4*4*4*64, num_classes, bias=True) # 8*8*8*64 -> 100 classes

    def forward(self, nodes, features, edges):
        '''Standard forward method for training on floats'''
        features = self.conv1(nodes, features, edges)
        features = self.relu1(features)
        features = self.conv2(nodes, features, edges)
        features = self.relu2(features)

        nodes, features, edges = self.max_pool1(nodes, features, edges)

        features = self.conv3(nodes, features, edges)
        features = self.relu3(features)
        features = self.conv4(nodes, features, edges)
        features = self.relu4(features)

        nodes, features, edges = self.max_pool2(nodes, features, edges)

        features = self.conv5(nodes, features, edges)
        features = self.relu5(features)
        features = self.conv6(nodes, features, edges)
        features = self.relu6(features)
        features = self.conv7(nodes, features, edges)
        features = self.relu7(features)

        features = self.out(nodes, features)
        features = self.dropout(features)
        return self.linear(features)
    
    def calibration(self, nodes, features, edges):
        '''Calibration method to adjust quantize parameters on dataset'''
        features = self.conv1.calibration(nodes, features, edges, use_obs=True)
        features = self.relu1.calibration(features)
        features = self.conv2.calibration(nodes, features, edges)
        features = self.relu2.calibration(features)
        nodes, features, edges = self.max_pool1(nodes, features, edges)

        features = self.conv3.calibration(nodes, features, edges, use_obs=True)
        features = self.relu3.calibration(features)
        features = self.conv4.calibration(nodes, features, edges)
        features = self.relu4.calibration(features)

        nodes, features, edges = self.max_pool2(nodes, features, edges)

        features = self.conv5.calibration(nodes, features, edges, use_obs=True)
        features = self.relu5.calibration(features)
        features = self.conv6.calibration(nodes, features, edges)
        features = self.relu6.calibration(features)
        features = self.conv7.calibration(nodes, features, edges)
        print(features)
        features = self.relu7.calibration(features)
        features = self.out.calibration(nodes, features)
        return self.linear.calibration(features)
    
    def freeze(self):
        '''Freeze parameters after calibration'''
        self.conv1.freeze()
        self.relu1.freeze(observer_in=self.conv1.observer_out)
        self.conv2.freeze(observer_in=self.conv1.observer_out)
        self.relu2.freeze(observer_in=self.conv2.observer_out)
        self.conv3.freeze(observer_in=self.conv2.observer_out)
        self.relu3.freeze(observer_in=self.conv3.observer_out)
        self.conv4.freeze(observer_in=self.conv3.observer_out)
        self.relu4.freeze(observer_in=self.conv4.observer_out)
        self.conv5.freeze(observer_in=self.conv4.observer_out)
        self.relu5.freeze(observer_in=self.conv5.observer_out)
        self.conv6.freeze(observer_in=self.conv5.observer_out)
        self.relu6.freeze(observer_in=self.conv6.observer_out)
        self.conv7.freeze(observer_in=self.conv6.observer_out)
        self.relu7.freeze(observer_in=self.conv7.observer_out)
        self.out.freeze(observer_in=self.conv7.observer_out)
        self.linear.freeze(observer_in=self.conv7.observer_out)

    def q_forward(self, nodes, features, edges):
        '''Forward method for quantized model'''
        features = self.conv1.q_forward(nodes, features, edges, first_layer=True)
        features = self.relu1.q_forward(features)
        features = self.conv2.q_forward(nodes, features, edges)
        features = self.relu2.q_forward(features)
        node, features, edge = self.max_pool1(nodes, features, edges)
        features = self.conv3.q_forward(node, features, edge)
        features = self.relu3.q_forward(features)
        features = self.conv4.q_forward(node, features, edge)
        features = self.relu4.q_forward(features)
        features = self.conv5.q_forward(node, features, edge)
        features = self.relu5.q_forward(features)
        node, features, edge = self.max_pool2(node, features, edge)
        features = self.conv6.q_forward(node, features, edge)
        features = self.relu6.q_forward(features)
        features = self.conv7.q_forward(node, features, edge)
        print(self.conv7.observer_out.dequantize_tensor(features))
        features = self.relu7.q_forward(features)
        features = self.out.q_forward(node, features)
        features = self.linear.q_forward(features)
        return self.linear.observer_out.dequantize_tensor(features)
    
    def get_parameters(self):
        #TODO - method for saving quantized parameters
        pass