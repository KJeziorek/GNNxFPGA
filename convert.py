import torch
import numpy as np
from models.qmodel import Model
from data.data_module import EventDM

from utils.quantize_model import post_training_quantization, quantize_aware_training
from utils.load_ckpt_model import load_ckpt_model

dm = EventDM(data_dir='dataset', data_name='ncaltech101', batch_size=1)
dm.setup()

model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8, bias=True)
model = load_ckpt_model(model, 'model.ckpt')
model = post_training_quantization(model, dm, num_calibration_samples=100, device='cuda')