import torch
import numpy as np
from models.qmodel import Model
from data.ncaltech101 import NCaltech101
from data.ncars import NCars

from utils.quantize_model import post_training_quantization, quantize_aware_training
from utils.load_ckpt_model import load_ckpt_model

torch.manual_seed(12345)


# dm = NCaltech101(data_dir='dataset', batch_size=1)
dm = NCars(data_dir='dataset', batch_size=1)
dm.setup()

model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8)
model = load_ckpt_model(model, 'ncars.ckpt')
model = post_training_quantization(model, dm, num_calibration_samples=500, device='cuda')


# model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8, bias=True)
# model = load_ckpt_model(model, 'model.ckpt')
# model = quantize_aware_training(model, dm, num_epochs=1, device='cuda')
