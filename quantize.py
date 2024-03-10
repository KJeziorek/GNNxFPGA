import torch
import numpy as np
from models.layers.graph_gen import GraphGen
from models.model import Model
from utils.normalise import normalise
from torch.nn import Module


f = open('/home/imperator/GNN/dataset/ncaltech101/train/accordion/image_0001.bin', 'rb')
raw_data = np.fromfile(f, dtype=np.uint8)
f.close()

raw_data = np.uint32(raw_data)

all_y = raw_data[1::5]
all_x = raw_data[0::5]
all_p = (raw_data[2::5] & 128) >> 7  # bit 7
all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
all_p = all_p.astype(np.float64)
all_p[all_p == 0] = -1

events = {}
events['x'] = all_x
events['y'] = all_y
events['t'] = all_ts
events['p'] = all_p

num_nodes = len(events['x'])
t = events['t'][num_nodes//2]
# Znalezienie indeksów używając numpy.searchsorted zamiast torch.searchsorted
index1 = np.clip(np.searchsorted(events['t'], t) - 1, 0, num_nodes - 1)
index0 = np.clip(np.searchsorted(events['t'], t-25000) - 1, 0, num_nodes - 1)

events['x'] = events['x'][index0:index1]
events['y'] = events['y'][index0:index1]
events['t'] = events['t'][index0:index1]
events['p'] = events['p'][index0:index1]

events = normalise(events)

graph_generator = GraphGen(3)

iter = 0
for event in events.astype(np.int32):
    graph_generator.forward(event)
    iter+=1
    if iter > 50:
        break
nodes, features, edges = graph_generator.release()

model = Model(256, bias=False, num_classes=100).to('cuda')

features = model(nodes, features, edges)
print(features)
print(model.conv1.linear.quant_weight())