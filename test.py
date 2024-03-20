import torch
import numpy as np
from models.tiny_model import Model
from data.ncaltech101 import NCaltech101
from data.ncars import NCars

from utils.quantize_model import post_training_quantization, quantize_aware_training, train_float_model
from utils.load_ckpt_model import load_ckpt_model

torch.manual_seed(12345)


# dm = NCaltech101(data_dir='dataset', batch_size=1)
dm = NCars(data_dir='dataset', batch_size=1, radius=3)
dm.setup()

model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8)
# model = train_float_model(model, dm, num_epochs=1, batch_size=16, device='cuda')
# torch.save(model.state_dict(), 'model_ncars.ckpt')
model.load_state_dict(torch.load('model_ncars.ckpt', map_location='cuda'))

# model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8)
# model = load_ckpt_model(model, 'ncars.ckpt')
# model = post_training_quantization(model, dm, num_calibration_samples=500, device='cuda')


# model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8)
# model = load_ckpt_model(model, 'ncars.ckpt')
model = quantize_aware_training(model, dm, num_epochs=1, device='cuda')
torch.save(model.state_dict(), 'model_quantized_ncars.ckpt')
