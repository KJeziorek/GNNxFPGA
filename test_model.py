from models.qmodel import Model
from models.layers.graph_gen import GraphGen
from utils.normalise import normalise
import numpy as np

ev = np.loadtxt('image_0003.txt').astype(np.float32)
events = {}
events['x'] = ev[:,0]
events['y'] = ev[:,1]
events['t'] = ev[:,2]
events['p'] = ev[:,3]

events = normalise(events)

graph_generator = GraphGen(3)
 
for event in events.astype(np.int32):
    if event[2] == 256:
        break
    graph_generator.forward(event)
nodes, features, edges = graph_generator.release()

model = Model()

model(nodes, features, edges)
model.eval()
print(model(nodes, features, edges))
print(model.calibration(nodes, features, edges))
model.freeze()
print(model.q_forward(nodes, features, edges))