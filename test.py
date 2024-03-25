import torch
import numpy as np
import os
from models.tiny_model import TModel
from models.base_model import BModel

from data.ncaltech101 import NCaltech101
from data.ncars import NCars
from data.mnistdvs import MnistDVS
from data.cifar10 import Cifar10

from utils.quantize_model import post_training_quantization, quantize_aware_training, train_float_model
from utils.load_ckpt_model import load_ckpt_model

torch.manual_seed(12345)


folder_name = 'results/tiny_model_mnist'
os.makedirs(folder_name, exist_ok=True)
# dm = NCaltech101(data_dir='dataset', batch_size=1, radius=3)
# dm = NCars(data_dir='dataset', batch_size=1, radius=3)
# dm = Cifar10(data_dir='dataset', batch_size=1, radius=3)
dm = MnistDVS(data_dir='dataset', batch_size=1, radius=3)
dm.setup()

model = TModel(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8, bias=False)
# model = train_float_model(model, dm, num_epochs=50, batch_size=64, device='cuda', dir_name=folder_name)

model.load_state_dict(torch.load(folder_name+'/float_model.ckpt', map_location='cuda'))

# model = post_training_quantization(model, dm, device='cuda', dir_name=folder_name)

# model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8)
# model = load_ckpt_model(model, 'ncars.ckpt')

# model = post_training_quantization(model, dm, device='cuda', dir_name=folder_name)
# torch.save(model.state_dict(), 'tiny_model_ptq_ncars.ckpt')


# model = Model(input_dimension=dm.dim, num_classes=dm.num_classes, num_bits=8)
# model = load_ckpt_model(model, 'ncars.ckpt')
model = quantize_aware_training(model, dm, num_epochs=20, batch_size=64, device='cuda', dir_name=folder_name)

# model.get_parameters()