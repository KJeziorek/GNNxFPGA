import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy
from tqdm import tqdm



def float_inference(model: nn.Module, 
                    data_loader: torch.utils.data.DataLoader,
                    device: str = 'cuda'):
    
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(data_loader)):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model(nodes, features, edges) # Float forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])

    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')

    return preds, y_true


def calibration_inference(model, 
                          data_loader: torch.utils.data.DataLoader,
                          num_calibration_samples: int = 500,
                          device: str = 'cuda'):
    
    for idx, batch in tqdm(enumerate(data_loader)):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        _ = model.calibration(nodes, features, edges) # Calibration forward pass
        if idx > num_calibration_samples:
            break
    return model

def quantize_inference(model: nn.Module, 
                    data_loader: torch.utils.data.DataLoader,
                    device: str = 'cuda'):
    preds = []
    y_true = []
    for idx, batch in tqdm(enumerate(data_loader)):
        nodes = batch['nodes'].to(device)
        features = batch['features'].to(device)
        edges = batch['edges'].to(device)
        pred = model.q_forward(nodes, features, edges) # Quantized forward pass
        y_pred = torch.argmax(pred, dim=-1)
        preds.append(y_pred.cpu().unsqueeze(0))
        y_true.append(batch['y'])
    
    preds = torch.cat(preds, dim=0).to('cpu')
    y_true = torch.tensor(y_true).to('cpu')

    return preds, y_true

def post_training_quantization(model: nn.Module, 
                               dm: L.LightningDataModule,
                               device: str = 'cuda',
                               dir_name: str = 'tiny_model'):
    
    accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)

    '''Post-training quantization of a model'''
    model = model.to(device)
    model.eval()

    print("\nRunning float model...")
    '''Performe evaluation on the validation data for float model'''
    preds, y_true = float_inference(model=model, data_loader=dm.val_dataloader(), device=device)
    print("\nAccuracy for float model on val dataset:", accuracy(preds, y_true).item())

    '''Performe evaluation on the test data for float model'''
    preds, y_true = float_inference(model=model, data_loader=dm.test_dataloader(), device=device)
    print("Accuracy for float model on test dataset:", accuracy(preds, y_true).item())

    '''Calibrate the model on the training data to determine the quantization parameters'''
    print("\nCalibrating model...")
    model = calibration_inference(model=model, data_loader=dm.val_dataloader(), num_calibration_samples=500, device=device)

    model.freeze() # Freeze the model for quantization

    '''Performe evaluation on the validation data'''
    print("\nRunning quantized model...")

    preds, y_true = quantize_inference(model=model, data_loader=dm.train_dataloader(), device=device)
    print("\nAccuracy for PTQ on train dataset:", accuracy(preds, y_true).item())

    preds, y_true = quantize_inference(model=model, data_loader=dm.val_dataloader(), device=device)
    print("\nAccuracy for PTQ on val dataset:", accuracy(preds, y_true).item())

    '''Performe evaluation on the test data'''
    preds, y_true = quantize_inference(model=model, data_loader=dm.test_dataloader(), device=device)
    print("Accuracy for PTQ on test dataset:", accuracy(preds, y_true).item())

    torch.save(model.state_dict(), dir_name+'ptq_model.ckpt')
    return model

def quantize_aware_training(model: nn.Module, 
                               dm: L.LightningDataModule,
                               num_epochs: int = 10,
                               device: str = 'cuda',
                               dir_name: str = 'tiny_model'):
    
    accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)

    '''First post-training quantization of a model'''
    model = model.to(device)
    model.eval()

    '''Performe evaluation on the validation data for float model'''
    print("\nRunning float model...")
    preds, y_true = float_inference(model=model, data_loader=dm.val_dataloader(), device=device)
    print("\nAccuracy for float model on val dataset:", accuracy(preds, y_true).item())

    '''Performe evaluation on the test data for float model'''
    preds, y_true = float_inference(model=model, data_loader=dm.test_dataloader(), device=device)
    print("Accuracy for float model on test dataset:", accuracy(preds, y_true).item())

    '''Calibrate the model on the training data to determine the quantization parameters'''
    print("\nCalibrating model...")
    model = calibration_inference(model=model, data_loader=dm.val_dataloader(), num_calibration_samples=500, device=device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
    model.train()

    '''Quantize-aware training'''
    print("\nQuantize-aware training...")
    for i in range(num_epochs):
        vec_pred = []
        vec_true = []
        print("Epoch:", i+1)
        for idx, batch in tqdm(enumerate(dm.train_dataloader())):
            nodes = batch['nodes'].to(device)
            features = batch['features'].to(device)
            edges = batch['edges'].to(device)
            optimizer.zero_grad()
            pred = model.calibration(nodes, features, edges)
            loss = criterion(pred, target=torch.tensor(batch['y']).long().to('cuda'))
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(pred, dim=-1)
            vec_pred.append(y_pred.cpu().unsqueeze(0))
            vec_true.append(batch['y'])
        
        vec_pred = torch.cat(vec_pred, dim=0).to('cpu')
        vec_true = torch.tensor(vec_true).to('cpu')
        print("Accuracy for QAT on train dataset:", accuracy(vec_pred, vec_true).item())

        model.eval()
        vec_pred = []
        vec_true = []
        for idx, batch in tqdm(enumerate(dm.val_dataloader())):
            nodes = batch['nodes'].to(device)
            features = batch['features'].to(device)
            edges = batch['edges'].to(device)
            pred = model.calibration(nodes, features, edges)

            y_pred = torch.argmax(pred, dim=-1)
            vec_pred.append(y_pred.cpu().unsqueeze(0))
            vec_true.append(batch['y'])
        
        vec_pred = torch.cat(vec_pred, dim=0).to('cpu')
        vec_true = torch.tensor(vec_true).to('cpu')
        print("Accuracy for QAT on val dataset:", accuracy(vec_pred, vec_true).item())

        vec_pred = []
        vec_true = []
        for idx, batch in tqdm(enumerate(dm.test_dataloader())):
            nodes = batch['nodes'].to(device)
            features = batch['features'].to(device)
            edges = batch['edges'].to(device)
            pred = model.calibration(nodes, features, edges)

            y_pred = torch.argmax(pred, dim=-1)
            vec_pred.append(y_pred.cpu().unsqueeze(0))
            vec_true.append(batch['y'])
        
        vec_pred = torch.cat(vec_pred, dim=0).to('cpu')
        vec_true = torch.tensor(vec_true).to('cpu')
        print("Accuracy for QAT on test dataset:", accuracy(vec_pred, vec_true).item())

    model.eval()
    model.freeze()

    '''Performe evaluation on the validation data'''
    vpreds, vy_true = quantize_inference(model=model, data_loader=dm.val_dataloader(), device=device)
    print("Accuracy for QAT model on val dataset:", accuracy(vpreds, vy_true).item())

    tpreds, ty_true = quantize_inference(model=model, data_loader=dm.test_dataloader(), device=device)
    print("Accuracy for QAT model on test dataset:", accuracy(tpreds, ty_true).item())

    torch.save(model.state_dict(), dir_name + '/qat_model.ckpt')
    return model


def train_float_model(model: nn.Module, 
                        dm: L.LightningDataModule,
                        num_epochs: int = 10,
                        batch_size: int = 16,
                        device: str = 'cuda',
                        dir_name: str = 'tiny_model'):
    
    best_test_accuracy = 0
    best_val_accuracy = 0

    accuracy = Accuracy(task="multiclass", num_classes=dm.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
    
    model.to(device)
    

    '''Float model training'''
    print("\nFloat model training...")
    for i in range(num_epochs):
        model.train()
        preds = []
        y_true = []
        print("Epoch:", i+1)
        for idx, batch in tqdm(enumerate(dm.train_dataloader())):
            nodes = batch['nodes'].to(device)
            features = batch['features'].to(device)
            edges = batch['edges'].to(device)
            pred = model(nodes, features, edges)
            loss = criterion(pred, target=torch.tensor(batch['y']).long().to('cuda'))
            
            loss = loss / batch_size
            loss.backward()

            if (idx+1) % batch_size == 0 or (idx+1) == len(dm.train_dataloader()):
                optimizer.step()
                optimizer.zero_grad()

            y_pred = torch.argmax(pred, dim=-1)
            preds.append(y_pred.cpu().unsqueeze(0))
            y_true.append(batch['y'])
        
        preds = torch.cat(preds, dim=0).to('cpu')
        y_true = torch.tensor(y_true).to('cpu')
        print("Accuracy for float model on train dataset:", accuracy(preds, y_true).item())

        model.eval()

        vpreds, vy_true = float_inference(model=model, data_loader=dm.val_dataloader(), device=device)
        print("Accuracy for float model on val dataset:", accuracy(vpreds, vy_true).item())

        tpreds, ty_true = float_inference(model=model, data_loader=dm.test_dataloader(), device=device)
        print("Accuracy for float model on test dataset:", accuracy(tpreds, ty_true).item())

        if accuracy(tpreds, ty_true).item() > best_test_accuracy or accuracy(vpreds, vy_true).item() > best_val_accuracy:
            best_test_accuracy = accuracy(tpreds, ty_true).item()
            best_val_accuracy = accuracy(vpreds, vy_true).item()
            torch.save(model.state_dict(), dir_name+'/float_model.ckpt')
    return model