import torch
from torch.nn import Module, Dropout
from torch.nn.functional import dropout, elu, relu
from models.layers.max_pool import GraphPooling
from models.layers.conv import GraphConv
from models.layers.pool_out import GraphPoolOut
    
class Model(Module):
    def __init__(self, input_dimension, bias=False, num_classes=100):
        super(Model, self).__init__()

        self.conv1 = GraphConv(input_dim=1, output_dim=8, bias=bias)
        # self.norm1 = torch.nn.BatchNorm1d(8)
        self.conv2 = GraphConv(input_dim=8, output_dim=16, bias=bias)
        # self.norm2 = torch.nn.BatchNorm1d(16)
        self.max_pool1 = GraphPooling(pool_size=4, max_dimension=input_dimension, only_vertices=False, self_loop=True)

        self.conv3 = GraphConv(input_dim=16, output_dim=32, bias=bias)
        # self.norm3 = torch.nn.BatchNorm1d(32)
        self.conv4 = GraphConv(input_dim=32, output_dim=32, bias=bias)
        # self.norm4 = torch.nn.BatchNorm1d(32)

        self.conv5 = GraphConv(input_dim=32, output_dim=64, bias=bias)
        # self.norm5 = torch.nn.BatchNorm1d(64)

        self.max_pool2 = GraphPooling(pool_size=4, max_dimension=64)

        self.conv6 = GraphConv(input_dim=64, output_dim=64, bias=bias)
        # self.norm6 = torch.nn.BatchNorm1d(64)
        self.conv7 = GraphConv(input_dim=64, output_dim=64, bias=bias)
        # self.norm7 = torch.nn.BatchNorm1d(64)

        # self.conv8 = GraphConv(input_dim=64, output_dim=128, bias=bias)
        # self.norm8 = torch.nn.BatchNorm1d(128)

        # self.max_pool3 = GraphPooling(pool_size=2, max_dimension=input_dimension//8, only_vertices=False, self_loop=True)

        # self.conv9 = GraphConv(input_dim=64, output_dim=64, bias=bias)
        # self.conv10 = GraphConv(input_dim=64, output_dim=64, bias=bias)
        # self.conv11 = GraphConv(input_dim=64, output_dim=64, bias=bias)

        self.out = GraphPoolOut(pool_size=4, max_dimension=16)
        self.dropout = Dropout(p=0.5)
        self.linear = torch.nn.Linear(4*4*4*64, num_classes, bias=True) # 8*8*8*64 -> 100 classes

    def forward(self, node, features, edges):
        # features = relu(self.conv1(node, features, edges))
        # features = self.norm1(features)
        # features = relu(self.conv2(node, features, edges))
        # features = self.norm2(features)

        # node, features, edges = self.max_pool1(node, features, edges)
        # # x = features.clone()
        # features = relu(self.conv3(node, features, edges))
        # features = self.norm3(features)
        # features = relu(self.conv4(node, features, edges))
        # features = self.norm4(features)

        # # features = features + x
        # features = relu(self.conv5(node, features, edges))
        # features = self.norm5(features)

        # node, features, edges = self.max_pool2(node, features, edges)
       
        # # x = features.clone()
        # features = relu(self.conv6(node, features, edges))
        # features = self.norm6(features)
        # features = relu(self.conv7(node, features, edges))
        # features = self.norm7(features)

        features = self.conv1(node, features, edges)
        features = self.conv2(node, features, edges)
        node, features, edges = self.max_pool1(node, features, edges)
        features = self.conv3(node, features, edges)
        features = self.conv4(node, features, edges)
        features = self.conv5(node, features, edges)
        node, features, edges = self.max_pool2(node, features, edges)
        features = self.conv6(node, features, edges)
        features = self.conv7(node, features, edges)

        # features = features + x

        # features = elu(self.conv8(node, features, edges))
        # features = self.norm8(features)

        features = self.out(node, features)
        features = self.dropout(features)
        return self.linear(features)