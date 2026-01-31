import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
import numpy as np
from ResNet8 import resnet8
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

class CustomHDF5Dataset(Dataset):
    def __init__(self, file_list, target_size=(254, 2), transform=None):
        self.file_list = file_list
        self.target_size = target_size
        self.transform = transform

        self.file_paths = []
        self.labels = []
        with open(file_list, 'r') as f:
            for line in f:
                file_path = line.strip()
                self.file_paths.append(file_path)
                if 'pos' in os.path.basename(file_path):
                    self.labels.append(1)
                else:
                    self.labels.append(0)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as f:
            data = torch.tensor(f['tensor'][:], dtype=torch.float32)
            label = self.labels[idx]

        data = self.pad_or_crop(data, self.target_size)

        if self.transform:
            data = self.transform(data)

        return data, label

    def pad_or_crop(self, data, target_size):
        target_height, target_width = target_size
        height, width = data.shape

        if height > target_height:
            data = data[:target_height, :]
        elif height < target_height:
            padding = torch.zeros((target_height - height, width), dtype=torch.float32)
            data = torch.cat([data, padding], dim=0)

        if width > target_width:
            data = data[:, :target_width]
        elif width < target_width:
            padding = torch.zeros((target_height, target_width - width), dtype=torch.float32)
            data = torch.cat([data, padding], dim=1)

        return data

def custom_collate_fn(batch):
    data = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    max_len = max([d.shape[0] for d in data])
    target_width = 2

    padded_data = []
    for d in data:
        if d.shape[0] < max_len:
            padding = torch.zeros((max_len - d.shape[0], target_width), dtype=torch.float32)
            d = torch.cat([d, padding], dim=0)
        padded_data.append(d)

    padded_data = torch.stack(padded_data)
    return padded_data, labels


def test_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device).float()

            # outputs = model(inputs).squeeze()
            
            
            outputs = model(inputs)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                outputs = outputs[0]
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            true_labels.extend(labels.cpu().numpy())
            # print(outputs)
            predicted = (outputs > 0.5).float()
            predicted_labels.extend(predicted.cpu().numpy())
            # print(predicted_labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    mcc = matthews_corrcoef(true_labels, predicted_labels) 

    print(f'Test Loss: {avg_loss:.5f}')
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision:.5f}')
    print(f'Test Recall: {recall:.5f}')
    print(f'Test F1 Score: {f1:.5f}')
    print(f'Test MCC: {mcc:.5f}')
    with open('./moad_test_result.txt','a+') as f:
        i = 0
        while i<2000:
            f.write(str(true_labels[i]) + '  ' + str(predicted_labels[i]) + '\n')
            i += 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = resnet8().to(device)
model_path = './resnet_best.pth' 
model.load_state_dict(torch.load(model_path))
model.eval()
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()


test_file_list = '/root/GENN_AE/resnet/test_data_list.txt'
test_dataset = CustomHDF5Dataset(file_list=test_file_list)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True, num_workers=10)

criterion = nn.BCELoss().to(device)

test_model(model, test_loader, criterion)
