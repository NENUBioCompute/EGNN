import os
import torch
import h5py
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import random
import numpy as np
import dgl
from model0710 import AE  # 假设 AE 定义在 model 模块中
from create_hdf5 import create_graph
from tqdm import tqdm

def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-7)

def clamp(tensor):
    return torch.clamp(tensor, 0, 1)

def encoder(graph, model, device):
    model.eval()
    with torch.no_grad():
        rec_graph = graph.to(device, non_blocking=True)
        h = rec_graph.ndata['feat'].float().to(device, non_blocking=True)
        edge_index_pos = rec_graph.edges(etype='pos')
        edge_index_cov = rec_graph.edges(etype='cov')
        edge_index_hb = rec_graph.edges(etype='hb')
        coord = rec_graph.ndata['x'].float().to(device, non_blocking=True)

        coord = normalize(coord)
        coord = clamp(coord)
        h = normalize(h)
        h = clamp(h)

        h_pred, x_coord = model.encode(h, edge_index_pos, edge_index_cov, edge_index_hb, coord, edge_attr=None)
        
    return h_pred, x_coord

def create_hdf5(inpth, outpth, model, device):
    graph = create_graph(inpth)
    h_pred, x_coord = encoder(graph, model, device)
    h_pred_cpu = h_pred.cpu()
    x_coord_cpu = x_coord.cpu()
    result = torch.cat((h_pred_cpu, x_coord_cpu), dim=1)
    
    with h5py.File(outpth, 'w') as f:
        f.create_dataset('tensor', data=result.numpy())

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化并加载模型
model = AE(hidden_nf=256, K=1, device=device, act_fn=nn.SiLU(), n_layers=3, reg=1e-3, clamp=False).to(device)
# model.load_state_dict(torch.load('/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/ae0710_nocov.pth'))
model.load_state_dict(
    torch.load('/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/pth/CNN_model_0630.pth',
               map_location=torch.device('cpu'))
)

# 读取文件列表
p_list = []

with open('/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/ae_train_list0630.txt', 'r') as fp:
    for line in fp:
        p_list.append(line.strip('\n'))
print(len(p_list))

# 处理文件并保存结果
for i in tqdm(p_list):
    if 'neg' in i:
        try:
            create_hdf5(i, '/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/neg/' + i.split('/')[-1].split('.')[0] + '.h5', model, device)
        except Exception as e:
            with open('/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/wrong.txt', 'a+') as fpp:
                fpp.write(i + '\n')
                fpp.write(str(e) + '\n')
    else:
        try:
            create_hdf5(i, '/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/pos/' + i.split('/')[-1].split('.')[0] + '.h5', model, device)
        except Exception as e:
            with open('/mnt/sfs_turbo/tyt/GENN_AE/GENN_AE/wrong.txt', 'a+') as fpp:
                fpp.write(i + '\n')
                fpp.write(str(e) + '\n')
