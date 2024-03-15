import torch
import numpy as np
from models.qmodel import Model
from torchmetrics import Accuracy
from data.data_module import EventDM


chackpoint = torch.load('model.ckpt', map_location=torch.device('cpu'))
model = Model(num_bits=8)
model_float = Model(num_bits=8)
accuracy = Accuracy(task="multiclass", num_classes=100)

#delete model. from state_dict
# print(chackpoint['state_dict'].keys())
# print(model.state_dict().keys())
new_state_dict = {}
for k, v in chackpoint['state_dict'].items():
    name = k[6:]
    new_state_dict[name] = v

for k in model.state_dict().keys():
    if k in new_state_dict.keys(): 
        print("Updating parameters:", k)
        model.state_dict()[k].copy_(new_state_dict[k])
        model_float.state_dict()[k].copy_(new_state_dict[k])

    # if k start with linear., remove linear. from k
    elif k[0:7] == 'linear.':
        if k[7:] in new_state_dict.keys():
            print("Updating parameters:", k)
            model.state_dict()[k].copy_(new_state_dict[k[7:]])
            model_float.state_dict()[k].copy_(new_state_dict[k[7:]])

dm = EventDM(data_dir='dataset', data_name='ncaltech101', batch_size=1)
dm.setup()


model.eval()
model_float.eval()

print("\nCalibrating model...")
for idx, batch in enumerate(dm.val_dataloader()):
    nodes = batch['nodes']
    features = batch['features']
    edges = batch['edges']
    y = batch['y']
    pred_float = model_float(nodes, features, edges)
    pred = model.calibration(nodes, features, edges)
    y_pred = torch.argmax(pred, dim=-1)
    y_pred_float = torch.argmax(pred_float, dim=-1)

    print("Prediction quantized:", y_pred.item(), "Prediction float:", y_pred_float.item())
    
    if idx == 20:
        break


print()
model.freeze()

print("Running model")
# print(model.linear.bias)

# print(model.conv1.linear.weight.data)
# print(model.conv1.linear.bias)
# print(model.conv1.observer_in.scale, model.conv1.observer_in.zero_point)
# print(model.conv1.observer_w.scale, model.conv1.observer_w.zero_point)
# print(model.conv1.observer_out.scale, model.conv1.observer_out.zero_point)


for idx, batch in enumerate(dm.val_dataloader()):
    nodes = batch['nodes']
    features = batch['features']
    edges = batch['edges']
    y = batch['y']
    pred_float = model_float(nodes, features, edges)
    pred = model.q_forward(nodes, features, edges)
    y_pred = torch.argmax(pred, dim=-1)
    y_pred_float = torch.argmax(pred_float, dim=-1)

    print("Prediction quantized:", y_pred.item(), "Prediction float:", y_pred_float.item())