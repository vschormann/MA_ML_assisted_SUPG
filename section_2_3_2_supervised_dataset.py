import os
from dolfinx import io
from dolfinx import fem
from dolfinx import default_scalar_type
from dolfinx import mesh as msh
import ufl
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mpi4py import MPI

from utils.FEniCSx_PyTorch_interface import FEniCSx_PyTorch_interface
from utils.FEniCSx_solver import FEniCSx_solver

batch_size = 10

class Dataset_example_2_1(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir

    def __len__(self):
        lst = os.listdir(self.input_dir)
        return len([f for f in lst if f.endswith('.pt')])

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, f"x_{idx}.pt")
        input = torch.load(input_path) 

        target_path = os.path.join(self.target_dir, f"t_{idx}.pt")
        target = torch.load(target_path) 

        return {'x':input, 't':target}
    
def collate_fn(batch):
    X = [b['x'] for b in batch]
    T  = [b['t'] for b in batch]
    return X, T

train_dataset = Dataset_example_2_1(input_dir="data/example_2_1/training_set/inputs/", target_dir="data/example_2_1/training_set/target_values")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

test_dataset = Dataset_example_2_1(input_dir="data/example_2_1/test_set/inputs/", target_dir="data/example_2_1/test_set/target_values/")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)