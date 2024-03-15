import torch
import numpy as np
from models.layers.qconv import QuantGraphConv
from models.layers.qlinear import QuantLinear
from models.layers.qrelu import QuantReLU
from models.layers.qpool_out import QuantGraphPoolOut

from models.layers.max_pool import GraphPooling
from models.layers.graph_gen import GraphGen

from utils.normalise import normalise

torch.manual_seed(123)
bits = 8

conv1 = QuantGraphConv(input_dim=1, output_dim=8, num_bits=bits)
relu1 = QuantReLU(num_bits=bits)
conv2 = QuantGraphConv(input_dim=8, output_dim=16, num_bits=bits)
relu2 = QuantReLU(num_bits=bits)
max_pool1 = GraphPooling(pool_size=4, max_dimension=256, num_bits=bits, only_vertices=False, self_loop=True)
conv3 = QuantGraphConv(input_dim=16, output_dim=32, num_bits=bits)
relu3 = QuantReLU(num_bits=bits)
conv4 = QuantGraphConv(input_dim=32, output_dim=32, num_bits=bits)
relu4 = QuantReLU(num_bits=bits)
conv5 = QuantGraphConv(input_dim=32, output_dim=64, num_bits=bits)
relu5 = QuantReLU(num_bits=bits)
max_pool2 = GraphPooling(pool_size=16, max_dimension=256, num_bits=bits, only_vertices=False, self_loop=True)
conv6 = QuantGraphConv(input_dim=64, output_dim=64, num_bits=bits)
relu6 = QuantReLU(num_bits=bits)
conv7 = QuantGraphConv(input_dim=64, output_dim=64, num_bits=bits)
relu7 = QuantReLU(num_bits=bits)

out = QuantGraphPoolOut(pool_size=64, max_dimension=256, num_bits=bits)
linear = QuantLinear(input_dim=4*4*4*64, output_dim=100, num_bits=bits, bias=True)

ev = np.loadtxt('image_0003.txt').astype(np.float32)
events = {}
events['x'] = ev[:,0]
events['y'] = ev[:,1]
events['t'] = ev[:,2]
events['p'] = ev[:,3]

events['p'][events['p'] == -1] = -1

events = normalise(events)

graph_generator = GraphGen(3)
 
for event in events.astype(np.int32):
    if event[2] == 256:
        break
    graph_generator.forward(event)
nodes, features, edges = graph_generator.release()

# print(features)
################################################################################

x = conv1(nodes, features, edges)
x = relu1(x)
x = conv2(nodes, x, edges)
x = relu2(x)
node, x, edge = max_pool1(nodes, x, edges)
x = conv3(node, x, edge)
x = relu3(x)
x = conv4(node, x, edge)
x = relu4(x)
x = conv5(node, x, edge)
x = relu5(x)
node, x, edge = max_pool2(node, x, edge)
x = conv6(node, x, edge)
x = relu6(x)
x = conv7(node, x, edge)
x = relu7(x)
x = out(node, x)
x = linear(x)

conv1.eval()
conv2.eval()
conv3.eval()
conv4.eval()
conv5.eval()
conv6.eval()
conv7.eval()

x = conv1(nodes, features, edges)
x = relu1(x)
x = conv2(nodes, x, edges)
x = relu2(x)
node, x, edge = max_pool1(nodes, x, edges)
x = conv3(node, x, edge)
x = relu3(x)
x = conv4(node, x, edge)
x = relu4(x)
x = conv5(node, x, edge)
x = relu5(x)
node, x, edge = max_pool2(node, x, edge)
x = conv6(node, x, edge)
x = relu6(x)
x = conv7(node, x, edge)
x = relu7(x)
with open('conv1.txt', 'w') as f:
    for i in x:
        f.write(str(i) + '\n')
x = out(node, x)
x = linear(x)



x = conv1.calibration(nodes, features, edges, use_obs=True)
x = relu1.calibration(x)
x = conv2.calibration(nodes, x, edges)
x = relu2.calibration(x)

node, x, edge = max_pool1.calibration(nodes, x, edges)

x = conv3.calibration(node, x, edge)
x = relu3.calibration(x)
x = conv4.calibration(node, x, edge)
x = relu4.calibration(x)
x = conv5.calibration(node, x, edge)
x = relu5.calibration(x)

node, x, edge = max_pool2.calibration(node, x, edge)

x = conv6.calibration(node, x, edge)
x = relu6.calibration(x)
x = conv7.calibration(node, x, edge)
x = relu7.calibration(x)
with open('calibration.txt', 'w') as f:
    for i in x:
        f.write(str(i) + '\n')
x = out.calibration(node, x)
x = linear.calibration(x)

print("Quantization")
conv1.freeze()
relu1.freeze(observer_in=conv1.observer_out)
conv2.freeze(observer_in=conv1.observer_out)
relu2.freeze(observer_in=conv2.observer_out)

max_pool1.freeze(observer_in=conv2.observer_out)

conv3.freeze(observer_in=max_pool1.observer_out)
relu3.freeze(observer_in=conv3.observer_out)
conv4.freeze(observer_in=conv3.observer_out)
relu4.freeze(observer_in=conv4.observer_out)
conv5.freeze(observer_in=conv4.observer_out)
relu5.freeze(observer_in=conv5.observer_out)

max_pool2.freeze(observer_in=conv5.observer_out)

conv6.freeze(observer_in=max_pool2.observer_out)
relu6.freeze(observer_in=conv6.observer_out)
conv7.freeze(observer_in=conv6.observer_out)
relu7.freeze(observer_in=conv7.observer_out)
out.freeze(observer_in=conv7.observer_out)
linear.freeze(observer_in=conv7.observer_out)

x = conv1.q_forward(nodes, features, edges, first_layer=True)
x = relu1.q_forward(x)
x = conv2.q_forward(nodes, x, edges)
x = relu2.q_forward(x)
node, x, edge = max_pool1.q_forward(nodes, x, edges)
x = conv3.q_forward(node, x, edge)
x = relu3.q_forward(x)
x = conv4.q_forward(node, x, edge)
x = relu4.q_forward(x)
x = conv5.q_forward(node, x, edge)
x = relu5.q_forward(x)
node, x, edge = max_pool2.q_forward(node, x, edge)
x = conv6.q_forward(node, x, edge)
x = relu6.q_forward(x)
x = conv7.q_forward(node, x, edge)
x = relu7.q_forward(x)

fx = conv7.observer_in.dequantize_tensor(x)
with open('quant.txt', 'w') as f:
    for i in fx:
        f.write(str(i) + '\n')

x = out.q_forward(node, x)
x = linear.q_forward(x)

fx = linear.observer_out.dequantize_tensor(x)




# print("Conv1")
# conv1.get_parameters()
# print("COnv2")
# conv2.get_parameters()