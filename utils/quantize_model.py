import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from tqdm import tqdm

def post_training_quantization(model: nn.Module, 
                               dm: L.LightningDataModule,
                               num_calibration_samples: int = 100,
                               device: str = 'cuda'):
    
    accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)

    '''Post-training quantization of a model'''
    model = model.to(device)
    model.eval()


    '''Performe evaluation on the validation data for float model'''
    print("\nRunning float model...")
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.val_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model(nodes, features, edges) # Float forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("\nAccuracy for float model on val dataset:", accuracy(preds, y_true).item())


    '''Performe evaluation on the test data for float model'''
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.test_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model(nodes, features, edges)  # Float forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("Accuracy for float model on test dataset:", accuracy(preds, y_true).item())


    '''Calibrate the model on the training data to determine the quantization parameters'''
    print("\nCalibrating model...")
    for idx, batch in tqdm(enumerate(dm.train_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        _ = model.calibration(nodes, features, edges) # Calibration forward pass
        if idx > num_calibration_samples:
            break


    model.freeze() # Freeze the model for quantization


    '''Performe evaluation on the validation data'''
    print("\nRunning quantized model...")
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.val_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges) # Quantized forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("\nAccuracy for PTQ on val dataset:", accuracy(preds, y_true).item())


    '''Performe evaluation on the test data'''
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.test_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges)
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("Accuracy for PTQ on test dataset:", accuracy(preds, y_true).item())

    return model


def quantize_aware_training(model: nn.Module, 
                               dm: L.LightningDataModule,
                               num_epochs: int = 10,
                               device: str = 'cuda'):
    
    accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)

    '''First post-training quantization of a model'''
    model = model.to(device)
    model.eval()


    '''Performe evaluation on the validation data for float model'''
    print("\nRunning float model...")
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.val_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model(nodes, features, edges) # Float forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("\nAccuracy for float model on val dataset:", accuracy(preds, y_true).item())


    '''Performe evaluation on the test data for float model'''
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.test_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model(nodes, features, edges)  # Float forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("Accuracy for float model on test dataset:", accuracy(preds, y_true).item())


    '''Calibrate the model on the training data to determine the quantization parameters'''
    print("\nCalibrating model...")
    for idx, batch in tqdm(enumerate(dm.train_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        _ = model.calibration(nodes, features, edges)
        if idx > 100:
            break
    
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
    model.train()


    '''Quantize-aware training'''
    print("\nQuantize-aware training...")
    for i in range(num_epochs):
        print("Epoch:", i+1)
        for idx, batch in tqdm(enumerate(dm.train_dataloader())):
            nodes = batch['nodes'].to(device)
            features = batch['features'].to(device)
            edges = batch['edges'].to(device)
            model.calibration(nodes, features, edges)
            optimizer.zero_grad()
            pred = model.calibration(nodes, features, edges)
            loss = criterion(pred, target=torch.tensor(batch['y']).long().to('cuda'))
            loss.backward()
            optimizer.step()


    model.eval()
    model.freeze()


    '''Performe evaluation on the validation data'''
    print("\nRunning quantized model...")
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.val_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges) # Quantized forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("\nAccuracy for QAT on val dataset:", accuracy(preds, y_true).item())


    '''Performe evaluation on the test data'''
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(dm.test_dataloader())):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges)
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')
    print("\nAccuracy for QAT on test dataset:", accuracy(preds, y_true).item())

    return model