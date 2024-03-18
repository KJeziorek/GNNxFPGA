import torch
import lightning as L

from torchmetrics import Accuracy
from torchmetrics.classification import ConfusionMatrix

from typing import Dict, Tuple
from torch.nn.functional import softmax
# from models.model import Model
from models.qmodel import Model

import wandb
import numpy as np
import matplotlib.pyplot as plt
import io


class LNModel(L.LightningModule):
    def __init__(self, lr, weight_decay, num_classes, batch_size, input_dimension=256):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.num_classes = num_classes

        self.input_dimension = input_dimension
        self.model = Model(input_dimension=input_dimension, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        if num_classes > 3:
            self.accuracy_top_3 = Accuracy(task="multiclass", num_classes=num_classes, top_k=3).to(self.device)

        # self.comfmat = ConfusionMatrix(task='multiclass', num_classes=100)

        self.save_hyperparameters()

        self.val_pred = None
        self.train_pred = None
        self.pred = []
        self.target = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, data):
        x = self.model(data['nodes'], data['features'], data['edges'])
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=torch.tensor(batch['y']).long().to('cuda'))

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction.cpu().unsqueeze(0), target=torch.tensor([batch['y']]))

        self.log('train_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('train_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)
        
        self.train_pred = {'y': batch['y'], 'y_pred': y_prediction.cpu().numpy(), 'nodes': batch['nodes']}

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        
        loss = self.criterion(outputs, target=torch.tensor(batch['y']).long().to('cuda'))
        y_prediction = torch.argmax(outputs, dim=-1)

        accuracy = self.accuracy(preds=y_prediction.cpu().unsqueeze(0), target=torch.tensor([batch['y']]))
        self.log('val_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('val_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.val_pred = {'y': batch['y'], 'y_pred': y_prediction.cpu().numpy(), 'nodes': batch['nodes']}

        if self.num_classes > 3:
            pred = softmax(outputs, dim=-1)
            top_3 = self.accuracy_top_3(preds=pred.unsqueeze(0).to(self.device), target=torch.tensor([batch['y']]).to(self.device))
            self.log('val_acc_top_3', top_3, on_epoch=True, logger=True, batch_size=self.batch_size)
    
    def on_train_epoch_end(self):
        event_image = self.create_events_image(self.train_pred['nodes'], image_size=(self.input_dimension, self.input_dimension))
        self.logger.experiment.log({"events_train": [wandb.Image(event_image, caption=f'GT: {self.train_pred["y"]} Pred: {self.train_pred["y_pred"]}')]})

    def on_validation_epoch_end(self):
        event_image = self.create_events_image(self.val_pred['nodes'], image_size=(self.input_dimension, self.input_dimension))
        self.logger.experiment.log({"events_val": [wandb.Image(event_image, caption=f'GT: {self.val_pred["y"]} Pred: {self.val_pred["y_pred"]}')]})

    def test_step(self, batch, batch_idx):
        outputs = self.forward(data=batch)
        loss = self.criterion(outputs, target=torch.tensor(batch['y']).long().to('cuda'))

        y_prediction = torch.argmax(outputs, dim=-1)
        accuracy = self.accuracy(preds=y_prediction.cpu().unsqueeze(0), target=torch.tensor([batch['y']]))

        self.log('test_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_acc', accuracy, on_epoch=True, logger=True, batch_size=self.batch_size)

    def create_events_image(self, events, image_size=(256, 256)):
        # Przekształć tensor na numpy array
        events_np = events.cpu().numpy()
        x, y = events_np[:, 0], events_np[:, 1]

        # Stwórz obraz z eventów
        hist, _, _ = np.histogram2d(x, y, bins=image_size)
        hist[hist > 0] = 255
        return hist.T