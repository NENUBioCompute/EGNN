import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import warnings
import dgl
from sklearn.model_selection import train_test_split
from model0710 import AE
# from model0710_2layer import AE
from create_hdf5 import create_graph

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

class RandomGraphDataset(Dataset):
    def __init__(self, rec_paths):
        self.rec_paths = rec_paths

    def __len__(self):
        return len(self.rec_paths)

    def __getitem__(self, idx):
        rec_path = self.rec_paths[idx]
        rec_graph = create_graph(rec_path)
        return rec_graph

def collate_fn(batch):
    graphs = dgl.batch(batch)
    return graphs

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-7)

def clamp(tensor):
    return torch.clamp(tensor, 0, 1)

# epochs = 200
epochs = 30
learning_rate = 0.0005
batch_size = 32
seed = 3
log_interval = 10

torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_rec_paths = []
with open('/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/ae_train_list.txt', 'r') as f:
    for line in f:
        all_rec_paths.append(line.strip('\n'))

rec_paths = random.sample(all_rec_paths, 50000)


train_paths, val_paths = train_test_split(rec_paths, test_size=0.2, random_state=seed)

train_dataset = RandomGraphDataset(train_paths)
val_dataset = RandomGraphDataset(val_paths)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=10)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=10)

input_nf = 16
hidden_nf = 256
latent_nf = 1
autoencoder = AE(hidden_nf=hidden_nf, K=1, device=device, act_fn=nn.SiLU(), n_layers=3, reg=1e-3, clamp=False).to(device)
# autoencoder.load_state_dict(torch.load('./ae0612.pth'))

# bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss_h = 0
    total_loss_coord = 0
    for batch_idx, rec_graph in enumerate(train_loader):
        rec_graph = rec_graph.to(device, non_blocking=True)
        h = rec_graph.ndata['feat'].float().to(device, non_blocking=True)
        edge_index_pos = rec_graph.edges(etype='pos')
        edge_index_cov = rec_graph.edges(etype='cov')
        edge_index_hb = rec_graph.edges(etype='hb')
        coord = rec_graph.ndata['x'].float().to(device, non_blocking=True)

        coord = normalize(coord)
        coord = clamp(coord)
        
        h = normalize(h)
        h = clamp(h)
        optimizer.zero_grad()

        h_pred, coord_pred = model(h, edge_index_pos, edge_index_cov, edge_index_hb, coord)
        h_pred = torch.sigmoid(h_pred)
        coord_pred = torch.sigmoid(coord_pred)

        coord_pred = coord_pred.expand_as(coord)
        h_pred = clamp(h_pred)
        coord_pred = clamp(coord_pred)

        if not torch.all((h_pred >= 0) & (h_pred <= 1)):
            raise ValueError(f"h_pred contains values out of range [0, 1]: {h_pred}")
        if not torch.all((coord_pred >= 0) & (coord_pred <= 1)):
            raise ValueError(f"coord_pred contains values out of range [0, 1]: {coord_pred}")

        try:
            loss_h = mse_loss(h_pred, h)
            loss_coord = mse_loss(coord_pred, coord)
            loss = loss_h + loss_coord
            total_loss_h += loss_h.item()
            total_loss_coord += loss_coord.item()
        except Exception as e:
            print(f'Error in loss calculation: {e}')
            print(f'h_pred: {h_pred}')
            print(f'h: {h}')
            print(f'coord_pred: {coord_pred}')
            print(f'coord: {coord}')
            raise e

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        processed_samples = batch_idx * batch_size
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{processed_samples}/{len(train_loader.dataset)} ({100. * processed_samples / len(train_loader.dataset):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss_h = total_loss_h / len(train_loader)
    avg_loss_coord = total_loss_coord / len(train_loader)
    print(f'Epoch {epoch} Training - Avg Loss_h: {avg_loss_h:.6f}, Avg Loss_coord: {avg_loss_coord:.6f}')

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    total_loss_h = 0
    total_loss_coord = 0
    with torch.no_grad():
        for rec_graph in val_loader:
            rec_graph = rec_graph.to(device)
            h = rec_graph.ndata['feat'].float().to(device)
            edge_index_pos = rec_graph.edges(etype='pos')
            edge_index_cov = rec_graph.edges(etype='cov')
            edge_index_hb = rec_graph.edges(etype='hb')
            coord = rec_graph.ndata['x'].float().to(device)

            coord = normalize(coord)
            coord = clamp(coord)
            
            h = normalize(h)
            h = clamp(h)

            h_pred, coord_pred = model(h, edge_index_pos, edge_index_cov, edge_index_hb, coord)
           
            h_pred = torch.sigmoid(h_pred)
            coord_pred = torch.sigmoid(coord_pred)

            coord_pred = coord_pred.expand_as(coord)
            h_pred = clamp(h_pred)
            coord_pred = clamp(coord_pred)

            if not torch.all((h_pred >= 0) & (h_pred <= 1)):
                raise ValueError(f"h_pred contains values out of range [0, 1]: {h_pred}")
            if not torch.all((coord_pred >= 0) & (coord_pred <= 1)):
                raise ValueError(f"coord_pred contains values out of range [0, 1]: {coord_pred}")

            try:
                loss_h = mse_loss(h_pred, h)
                loss_coord = mse_loss(coord_pred, coord)
                loss = loss_h + loss_coord
                total_loss_h += loss_h.item()
                total_loss_coord += loss_coord.item()
            except Exception as e:
                print(f'Error in loss calculation: {e}')
                print(f'h_pred: {h_pred}')
                print(f'h: {h}')
                print(f'coord_pred: {coord_pred}')
                print(f'coord: {coord}')
                raise e

            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    avg_loss_h = total_loss_h / len(val_loader)
    avg_loss_coord = total_loss_coord / len(val_loader)
    print(f'Validation - Avg Loss_h: {avg_loss_h:.6f}, Avg Loss_coord: {avg_loss_coord:.6f}')
    
    return val_loss

best_val_loss = float('inf')
for epoch in range(1, epochs + 1):
    train(autoencoder, device, train_loader, optimizer, epoch)
    val_loss = validate(autoencoder, device, val_loader)
    print(f'Epoch {epoch}, Validation Loss: {val_loss:.6f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(autoencoder.state_dict(), '/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/ae_best.pth')
        print(f'Saved best model with loss {best_val_loss:.6f} at epoch {epoch}')

print('Training complete.')
