import os
import glob
import numpy as np
import torch
import lightning as L

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from torch.utils.data import random_split, DataLoader
from torch.utils.data.dataset import Dataset

from models.layers.graph_gen import GraphGen
from models.layers.augmentation import RandomHorizontalFlip, RandomPolarityFlip, RandomRotationEvent
from utils.normalise import normalise

device = torch.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')

class NCars(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = 'ncars'

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.dim = 128

        self.num_workers = 2
        self.batch_size = batch_size
        self.processes = 6

        self.num_classes = 2
        self.class_dict = {'background': 0, 'car': 1}

        self.random_flip = RandomHorizontalFlip(0.5)
        self.random_polarity_flip = RandomPolarityFlip(0.5)
        # self.random_rotation = RandomRotationEvent(5)

        self.augmentations = [self.random_flip, self.random_polarity_flip]

    def prepare_data(self) -> None:
        print('Preparing data...')
        for mode in ['train', 'val', 'test']:
            print(f'Loading {mode} data')
            os.makedirs(os.path.join(self.data_dir, self.data_name + '_processed', mode), exist_ok=True)
            self._prepare_data(mode)

    def _prepare_data(self, mode: str) -> None:
        data_files = glob.glob(os.path.join(self.data_dir, self.data_name, mode, '*', 'events.txt'))
        process_map(self.process_file, data_files, max_workers=self.processes, chunksize=1, )
            
    def process_file(self, data_file) -> None:   
        processed_file = data_file.replace(self.data_name, self.data_name + '_processed').replace('txt', 'pt')

        if os.path.exists(processed_file):
            return

        os.makedirs(os.path.dirname(processed_file), exist_ok=True)

        events_file = os.path.join(data_file)
        events = np.loadtxt(events_file)

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
        
        events = normalise(events, self.dim, x_max=120, y_max=100, t_max=events['t'].max())
        graph_generator = GraphGen(r=3, dimension_XY=self.dim, self_loop=True).to(device)

        for event in events.astype(np.int32):
            graph_generator.forward(event)
        nodes, features, edges = graph_generator.release()
        

        y = np.loadtxt(data_file.replace('events.txt', 'is_car.txt')).astype(np.int32)
        data = {'nodes': nodes.to("cpu"), 'features': features.to("cpu"), 'edges': edges.to("cpu"), 'y': y.item()}
        # Save processed file
        torch.save(data, processed_file)

    def setup(self, stage=None):
        self.train_data = self.generate_ds('train', self.augmentations)
        self.val_data = self.generate_ds('val')
        self.test_data = self.generate_ds('test')

    def generate_ds(self, mode: str, augmentations=None):
        processed_files = glob.glob(os.path.join(self.data_dir, self.data_name + '_processed',  mode, '*', '*.pt'))
        return EventDS(processed_files, augmentations, self.dim)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate_fn, persistent_workers=False)
    
    def collate_fn(self, data_list):
        # Load batch_size files into list and merged them to one big Data object
        # batch = Batch.from_data_list(data_list)
        # batch.batch_idx =torch.tensor(sum([[i] * len(data.y) for i, data in enumerate(data_list)], []))
        return data_list[0]
    
class EventDS(Dataset):
    def __init__(self, files, augmentations=None, dim=128):
        self.files = files
        self.augmentations = augmentations
        self.dim = dim

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int):
        data_file = self.files[index]
        data = torch.load(data_file)

        if self.augmentations:
            for aug in self.augmentations:
                data['nodes'], data['features'] = aug(data['nodes'], data['features'], self.dim)

        return data