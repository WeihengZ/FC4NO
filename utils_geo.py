from torch_geometric.data import Dataset, Data, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TensorDataLoader
import time
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(r'./')
from models.chamfer_distance.chamfer_distance import ChamferDistance

# Model training and testing
def train_model(args, model, optimizer, device, train_loader, val_loader, scale_factor, epochs=2):
    
    criterion = ChamferDistance()
    best_val_err = np.inf
    for epoch in range(epochs):

        if epoch % 2 == 1:
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
            }, './res/trained_model/geo_encoder/model_{}.pt'.format(args.data))

        model.train()
        num_samples = 0
        for data in train_loader:

            if np.random.rand() >= 0.3:
                continue

            if num_samples % 10 == 0:
                print(num_samples)
            
            # extract data
            graph_data = data.x.to(device).unsqueeze(0).permute(0,2,1)
            num_nodes = graph_data.shape[-1] 
            if num_nodes > 50000:
                graph_data = graph_data[:,:,np.random.choice(num_nodes, 50000)]

            # model optimization
            optimizer.zero_grad()
            output, _ = model(graph_data)
            loss = criterion(output.permute(0,2,1), graph_data.permute(0,2,1))

            loss.backward()
            optimizer.step()
            et = time.time()
            num_samples += 1
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.8f}")


def test_model(args, model, device, test_loader, scale_factor):
    model.eval()
    all_H = []
    num_samples = 0
    with torch.no_grad():
        for data in test_loader:

            if num_samples % 10 == 0:
                print(num_samples)

            # extract data
            graph_data = data.x.to(device).unsqueeze(0).permute(0,2,1)
            num_nodes = graph_data.shape[-1] 
            if num_nodes > 50000:
                graph_data = graph_data[:,:,np.random.choice(num_nodes, 50000)]

            output, H = model(graph_data)
            all_H.append(H.detach().cpu().numpy())
            num_samples += 1

    return all_H
