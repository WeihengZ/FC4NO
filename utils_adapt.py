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

def train_grid_model(args, sample_IDs, model, optimizer, device, data_loader, scale_factor, epochs=1):
    
    criterion = nn.MSELoss()
    best_val_err = np.inf
    for epoch in range(epochs):

        if epoch % 1 == 0:
            print('validating')
            val_err = validate_model(args, model,  device, data_loader, scale_factor)
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
        for graph_data, train_flag in data_loader:

            if train_flag == False:
                continue

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