import torch
from data.ncars import NCars
from models.tiny_model import Model
import numpy as np
from utils.normalise import normalise 
from utils.load_ckpt_model import load_ckpt_model
from models.layers.utils.quantize import quantize_tensor, dequantize_tensor

from models.layers.graph_gen import GraphGen

events = np.loadtxt('dataset/ncars/val/sequence_12961/events.txt')

all_x = events[:, 0]
all_y = events[:, 1]
all_ts = events[:, 2]
all_p = events[:, 3]
all_p[all_p == 0] = -1

events = {}
events['x'] = all_x
events['y'] = all_y
events['t'] = all_ts.astype(np.float64)
events['p'] = all_p

events = normalise(events, 128, x_max=120, y_max=100, t_max=events['t'].max())

graph_generator = GraphGen(r=3, dimension_XY=128, self_loop=True).to('cuda')

for event in events.astype(np.int32):
    graph_generator.forward(event)
nodes, features, edges = graph_generator.release()

model = Model(input_dimension=128, num_classes=2, num_bits=8).to('cuda')

model.calibration(nodes.to('cuda'), features.to('cuda'), edges.to('cuda'))
model.eval()
model.freeze()
                  
param = torch.load('model_quantized_ncars.ckpt', map_location='cuda')

for pa in param:
    model.state_dict()[pa].copy_(param[pa])
    print(pa)
    print(param[pa])
    print(model.state_dict()[pa])


model.get_parameters()
# print(model.conv1)
# _ = model.q_forward(nodes.to('cuda'), features.to('cuda'), edges.to('cuda'))

# model.get_parameters()
