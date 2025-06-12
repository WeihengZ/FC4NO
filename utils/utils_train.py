from torch_geometric.data import Dataset, Data, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TensorDataLoader
import time
import matplotlib.pyplot as plt
import os
import gc

def train_grid_model(args, model, optimizer, device, train_loader, val_loader, scale_factor, epochs=2):
    
    criterion = nn.MSELoss()
    best_val_err = np.inf
    for epoch in range(epochs):

        if epoch % 3 == 1:
            print('validating')
            val_err = validate_model(args, model,  device, val_loader, scale_factor)
            if val_err < best_val_err:
                best_val_err = val_err
                torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                }, './res/trained_model/{}/model_{}_{}.pt'.format(args.model, args.model, args.data))
            print('Current best validation errs by now:', best_val_err)

        model.train()
        total_loss = 0
        num = 0
        for graph_data, _ in train_loader:

            # extract data
            graph_data = graph_data.to(device)

            # model optimization
            optimizer.zero_grad()
            try:
                output = model(graph_data)
            except:
                print('OOM error')
                gc.collect()  
                torch.cuda.empty_cache()  
                continue

            # designed for trnsformer
            if isinstance(output, tuple):
                loss = criterion(output[0], output[1])
            else:
                loss = criterion(output, graph_data.y.to(device))
            

            loss.backward()
            optimizer.step()
            et = time.time()
            total_loss += loss.detach().cpu().item()

            # memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
            # memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
            # print(f"Batch {0}, Loss: {loss.item():.6f}, GPU Memory Allocated: {memory_allocated:.2f} MB, Reserved: {memory_reserved:.2f} MB")

            gc.collect()                      # Python garbage collection
            torch.cuda.empty_cache()          # Release cached memory back to CUDA driver
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.8f}")

# Model training and testing
def train_model(args, model, optimizer, device, train_loader, val_loader, scale_factor, epochs=2):
    
    criterion = nn.MSELoss()
    best_val_err = np.inf
    save_dir = './res/trained_model/{}/'.format(args.model)
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):

        if epoch % args.val_freq == 0:
            print('validating')
            val_err = validate_model(args, model,  device, val_loader, scale_factor)
            print('validation error this epoch:', val_err)
            if val_err < best_val_err:
                best_val_err = val_err
                torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                }, os.path.join(save_dir, 'model_{}_{}.pt'.format(args.model, args.data)))
                print('saved new model.')
            print('Current best validation errs by now:', best_val_err)

        model.train()
        for data, _ in train_loader:
            
            # extract data
            graph_data = data.to(device)

            # model optimization
            optimizer.zero_grad()
            output = model(graph_data)

            # designed for trnsformer
            if isinstance(output, tuple):
                loss = criterion(output[0], output[1])
            else:
                loss = criterion(output, graph_data.y)

            loss.backward()
            optimizer.step()
            et = time.time()

            gc.collect()                      # Python garbage collection
            torch.cuda.empty_cache()          # Release cached memory back to CUDA driver
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.8f}")

def validate_model(args, model, device, val_loader, scale_factor):
    model.eval()
    avg_stress_err = 0
    num_samples = 0
    with torch.no_grad():
        for data, _ in val_loader:
            
            # extract data
            graph_data = data.to(device)

            if hasattr(model, 'predict'):
                output = model.predict(graph_data)
            else:
                output = model(graph_data)

            if isinstance(output, torch.Tensor):
                pred = output.detach().cpu().numpy()
                GT = graph_data.y.detach().cpu().numpy()
            elif isinstance(output, tuple):
                pred = output[0].detach().cpu().numpy()
                GT = output[1].detach().cpu().numpy()
            else:
                pred = output
                GT = graph_data.y.detach().cpu().numpy()  

            # re-normalize
            try:
                mean_ = scale_factor['mean']
                std_ = scale_factor['std']
                GT = GT * (std_ ) + mean_
                pred = pred * (std_ ) + mean_
            except:
                max_ = scale_factor['max']
                min_ = scale_factor['min']
                GT = GT * (max_ - min_) + min_
                pred = pred * (max_ - min_) + min_
        
            if np.linalg.norm(GT, ord=2) > 0:
                stress_relative_err = np.linalg.norm(GT - pred, ord=2) / (np.linalg.norm(GT, ord=2))
                avg_stress_err += stress_relative_err
                num_samples += 1

    return avg_stress_err / num_samples

def test_model(args, model, device, test_loader, scale_factor):
    model.eval()
    avg_strain_err = 0
    avg_stress_err = 0
    num_samples = 0
    all_err = []
    with torch.no_grad():
        for data, _ in test_loader:

            # extract data
            graph_data = data.to(device)

            if hasattr(model, 'predict'):
                output = model.predict(graph_data)
            else:
                output = model(graph_data)

            GT = graph_data.y.detach().cpu().numpy()
            if isinstance(output, torch.Tensor):
                pred = output.detach().cpu().numpy()
            else:
                pred = output

            # re-normalize
            try:
                mean_ = scale_factor['mean']
                std_ = scale_factor['std']
                GT = GT * (std_ ) + mean_
                pred = pred * (std_ ) + mean_
            except:
                max_ = scale_factor['max']
                min_ = scale_factor['min']
                GT = GT * (max_ - min_) + min_
                pred = pred * (max_ - min_) + min_
        
            if np.linalg.norm(GT, ord=2) > 0:
                stress_relative_err = np.linalg.norm(GT - pred, ord=2) / (np.linalg.norm(GT, ord=2))
                all_err.append(stress_relative_err)
                avg_stress_err += stress_relative_err
                num_samples += 1
    print(np.std(np.array(all_err)))
    print(np.mean(np.array(all_err)))
    print(all_err)

    print('evalute average performance on {} samples'.format(num_samples))
    print('average stess relative err:', avg_stress_err / num_samples)

