import os
import torch
from torch.utils.data import Dataset
from functools import lru_cache



class supervised_dataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, target_transform=None, device=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        lst = os.listdir(self.input_dir)
        return len([f for f in lst if f.startswith("x_") and  f.endswith('.pt')])
    

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, f"x_{idx}.pt")
        input = torch.load(input_path, map_location=self.device) 

        target_path = os.path.join(self.target_dir, f"t_{idx}.pt")
        target = torch.load(target_path, map_location=self.device) 

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target