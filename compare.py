from models.layers.graph_gen import GraphGen
from models.layers.max_pool import GraphPooling
from models.layers.pool_out import GraphPoolOut
from models.layers.conv import GraphConv

from models.model import Model
# from models.model_pyg import GNNModel


import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.normalise import normalise
from typing import Callable, List, Optional, Tuple, Union


# from models.layers.augmentation import RandomHorizontalFlip, RandomPoraityFlip, RandomRotationEvent

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
 
for event in events.astype(np.int32):
    graph_generator.forward(event)
nodes, features, edges = graph_generator.release()

model = Model(256, False, 100).to('cuda').eval()

features = model(nodes, features, edges)

# out = GraphPoolOut(64, 256)
# output_features = out(nodes, features)
# print(output_features)
# flip = RandomHorizontalFlip(1)
# rotate = RandomRotationEvent(5)
# pol_flip = RandomPoraityFlip(1)

# # VIsualize nodes in 2D image
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(nodes[:, 0].cpu(), nodes[:, 1].cpu(), c=features.cpu(), s=1, cmap='bwr')
# plt.show(block=False)

# nodesz, featuresz = rotate(nodes, features, 256)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(nodesz[:, 0].cpu(), nodesz[:, 1].cpu(), c=featuresz.cpu(), s=1, cmap='bwr')
# plt.show(block=False)

# nodesz, featuresz = flip(nodes, features, 256)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(nodesz[:, 0].cpu(), nodesz[:, 1].cpu(), c=featuresz.cpu(), s=1, cmap='bwr')
# plt.show(block=False)

# nodesz, featuresz = pol_flip(nodes, features, 256)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(nodesz[:, 0].cpu(), nodesz[:, 1].cpu(), c=featuresz.cpu(), s=1, cmap='bwr')
# plt.show()

# own_pool_out = GraphPoolOut(64, 256)
# pool_out = MaxPoolingX([64, 64], 4*4)
# conv = GraphConv(1, 4).cuda()

# features = conv(nodes, features, edges)

# out_own = own_pool_out(nodes, features)
# out = pool_out(features, nodes[:, :2])

# print(out_own)
# print(out)

# own_pool = GraphPooling(8, 256, False, False)
# pyg_pool = MaxPooling([4, 4])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(nodes[:, 0].cpu(), nodes[:, 1].cpu(), nodes[:, 2].cpu(), c=features.cpu(), s=1, cmap='bwr')
# plt.show(block=False)

# print(nodes.size())
# print(edges.size())
# nodes_, features_, edges_ = own_pool(nodes, features, edges)
# nodes_, features_, edges_ = nodes_.cpu(), features_.cpu(), edges_.cpu()

# print(edges_)
# print(nodes_.size())
# print(edges_.size())
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(nodes_[:, 0], nodes_[:, 1], nodes_[:, 2], c=features_, s=1, cmap='bwr')
# # for edge in edges_:
# #     ax.plot([nodes_[edge[0], 0], nodes_[edge[1], 0]], [nodes_[edge[0], 1], nodes_[edge[1], 1]], [nodes_[edge[0], 2], nodes_[edge[1], 2]], c='black')
# plt.show()

# nodes_, features_, edges_ = pyg_pool(nodes, features, edges)
# nodes_, features_, edges_ = nodes_.cpu(), features_.cpu(), edges_.cpu()

# print(edges_)
# print(nodes_.size())
# print(edges_.size())
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(nodes_[:, 0], nodes_[:, 1], nodes_[:, 2], c=features_, s=1, cmap='bwr')
# # for edge in edges_:
# #     ax.plot([nodes_[edge[0], 0], nodes_[edge[1], 0]], [nodes_[edge[0], 1], nodes_[edge[1], 1]], [nodes_[edge[0], 2], nodes_[edge[1], 2]], c='black')
# # plt.show()

