import torch
from data.ncars import NCars
from models.tiny_model import TModel
import numpy as np
from utils.normalise import normalise 
from utils.load_ckpt_model import load_ckpt_model
from models.layers.utils.quantize import quantize_tensor, dequantize_tensor

from utils.quantize_model import quantize_inference, float_inference
from models.layers.graph_gen import GraphGen
from torchmetrics import Accuracy

events = np.loadtxt('dataset/ncars/train/sequence_12961/events.txt')

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

events = normalise(events, 128, x_max=120, y_max=100, t_max=0.1)

graph_generator = GraphGen(r=3, dimension_XY=128, self_loop=True).to('cuda')

for event in events.astype(np.int32):
    graph_generator.forward(event)
nodes, features, edges = graph_generator.release()

model = TModel(input_dimension=128, num_classes=2, num_bits=8).to('cuda')

# model.eval()
model.calibration(nodes.to('cuda'), features.to('cuda'), edges.to('cuda'))

model.eval()
model.freeze()

param = torch.load('qat_model.ckpt', map_location='cuda')
for pa in param:
    model.state_dict()[pa].copy_(param[pa])
# model.load_state_dict(torch.load('tiny_model_ncars/float_model.ckpt', map_location='cuda'))

# model.get_parameters()
model.q_forward(nodes.to('cuda'), features.to('cuda'), edges.to('cuda'))

# dm = NCars(data_dir='dataset', batch_size=1, radius=3)
# dm.setup()
# accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)

# pred, true = float_inference(model, dm.train_dataloader(), 'cuda')
# print("Accuracy for float model on train dataset:", accuracy(pred, true).item())

# pred, true = float_inference(model, dm.val_dataloader(), 'cuda')
# print("Accuracy for float model on val dataset:", accuracy(pred, true).item())

# pred, true = float_inference(model, dm.test_dataloader(), 'cuda')
# print("Accuracy for float model on test dataset:", accuracy(pred, true).item())












# model.calibration(nodes.to('cuda'), features.to('cuda'), edges.to('cuda'))
# model.eval()
# model.freeze()
                  
# param = torch.load('tiny_model_ncars/qat_model.ckpt', map_location='cuda')

# for pa in param:
#     model.state_dict()[pa].copy_(param[pa])

# model.q_forward(nodes.to('cuda'), features.to('cuda'), edges.to('cuda'))



















# pred, true = quantize_inference(model, dm.train_dataloader(), 'cuda')
# print("Accuracy for float model on train dataset:", accuracy(pred, true).item())

# pred, true = quantize_inference(model, dm.val_dataloader(), 'cuda')
# print("Accuracy for float model on val dataset:", accuracy(pred, true).item())

# pred, true = quantize_inference(model, dm.test_dataloader(), 'cuda')
# print("Accuracy for float model on test dataset:", accuracy(pred, true).item())