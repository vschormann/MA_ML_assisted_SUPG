import torch.nn.functional as F
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

def train(model, loader, optimizer, device):
    #model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = F.mse_loss(out, data.y.view(-1,1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


class graph_dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith(".pt")]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list = []

        for raw_path in self.raw_paths:
            data = torch.load(raw_path, weights_only=False)
            
            assert data.x is not None
            assert data.edge_index is not None

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

train_set = graph_dataset(root="data/training_set/input_values/")
test_set = graph_dataset(root="data/test_set/input_values/")

class train_loader(DataLoader):
    def __init__(self, batch_size):
        super().__init__(graph_dataset(root="data/training_set/input_values/"), batch_size=batch_size)

class test_loader(DataLoader):
    def __init__(self, batch_size):
        super().__init__(graph_dataset(root="data/test_set/input_values/"), batch_size=batch_size)