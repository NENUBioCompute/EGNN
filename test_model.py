import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class CustomHDF5Dataset(Dataset):
    def __init__(self, file_list, target_size=(180, 2), transform=None):
        self.file_list = file_list
        self.target_size = target_size
        self.transform = transform

        self.file_paths = []
        self.labels = []
        with open(file_list, 'r') as f:
            for line in f:
                if 'pos' in line:
                    file_path = '/root/autodl-tmp/CNN_data/pos/' + line.strip()  # Ignore the second column
                else:
                    file_path = '/root/autodl-tmp/CNN_data/neg/' + line.strip()
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
    target_width = 2  # 固定宽度为2

    padded_data = []
    for d in data:
        if d.shape[0] < max_len:
            padding = torch.zeros((max_len - d.shape[0], target_width), dtype=torch.float32)
            d = torch.cat([d, padding], dim=0)
        padded_data.append(d)

    padded_data = torch.stack(padded_data)
    return padded_data, labels


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 2), stride=1, padding=(2, 0))  # 增加输出通道数到32
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1), stride=1, padding=(2, 0))  # 增加输出通道数到64
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(64 * 45 * 1, 256)  # 增加全连接层的神经元数量到256
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 45 * 1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.to(device)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test code
test_file_list = './test_list.txt'
test_dataset = CustomHDF5Dataset(file_list=test_file_list)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

net = CustomCNN().to(device)
load_model(net, './ae_best.pth')

criterion = nn.BCELoss()

test_loss, test_accuracy = evaluate_model(net, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_accuracy:.2f}%')
