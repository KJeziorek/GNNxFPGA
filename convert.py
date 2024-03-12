import torch
import numpy as np



param = torch.load('model.ckpt')
print(param['state_dict']['model.conv1.linear.weight'])