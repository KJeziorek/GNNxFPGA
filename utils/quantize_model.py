import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy

def post_training_quantization(model: nn.Module, 
                               dm: L.LightningDataModule,
                               num_calibration_samples: int = 100,
                               device: str = 'cuda'):
    
    accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)

    '''Post-training quantization of a model'''
    model = model.to(device)
    model.eval()

    '''Calibrate the model on the training data to determine the quantization parameters'''
    print("\nCalibrating model...")
    for idx, batch in enumerate(dm.train_dataloader()):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        model.calibration(nodes, features, edges)
        if idx > num_calibration_samples:
            break

    model.freeze()

    '''Performe evaluation on the validation data'''
    print("\nRunning model on validation dataset...")
    preds = []
    y_true = []
    for idx, batch in enumerate(dm.val_dataloader()):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges)
        preds.append(pred.unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')

    print("Accuracy for PTQ on val dataset:", accuracy(preds, y_true))

    '''Performe evaluation on the test data'''
    print("\nRunning model on test dataset...")
    preds = []
    y_true = []
    for idx, batch in enumerate(dm.test_dataloader()):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges)
        preds.append(pred.unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')

    print("Accuracy for PTQ on test dataset:", accuracy(preds, y_true))

    return model


def quantize_aware_training(model: nn.Module, 
                               dm: L.LightningDataModule,
                               num_epochs: int = 10,
                               device: str = 'cuda'):
    
    accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)

    '''First post-training quantization of a model'''
    model = model.to(device)
    model.eval()

    '''Calibrate the model on the training data to determine the quantization parameters'''
    print("\nCalibrating model...")
    for idx, batch in enumerate(dm.train_dataloader()):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        model.calibration(nodes, features, edges)
        if idx > 100:
            break
    
    model.train()

    '''Quantize-aware training'''
    print("\nQuantize-aware training...")
    for i in range(num_epochs):
        print("Epoch:", i+1)
        for idx, batch in enumerate(dm.train_dataloader()):
            nodes = batch['nodes'].to(device)
            features = batch['features'].to(device)
            edges = batch['edges'].to(device)
            model.calibration(nodes, features, edges)
            if idx > 100:
                break
            optimizer.zero_grad()
            pred = model.calibration(nodes, features, edges)
            loss = criterion(pred, target=torch.tensor(batch['y']).long().to('cuda'))
            loss.backward()
            optimizer.step()


    model.freeze()

    '''Performe evaluation on the validation data'''
    print("\nRunning model on validation dataset...")
    preds = []
    y_true = []
    for idx, batch in enumerate(dm.val_dataloader()):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges)
        preds.append(pred.unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')

    print("Accuracy for PTQ on val dataset:", accuracy(preds, y_true))

    '''Performe evaluation on the test data'''
    print("\nRunning model on test dataset...")
    preds = []
    y_true = []
    for idx, batch in enumerate(dm.test_dataloader()):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges)
        preds.append(pred.unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')

    print("Accuracy for PTQ on test dataset:", accuracy(preds, y_true))

    return model