import torch

class flattening_transform:
    def __init__(self, flat_key: torch.Tensor, C: int):
        self.flat_key = flat_key
        self.C = C

    def __call__(self, x: torch.Tensor):
        x = x.reshape(self.C,-1)[:,self.flat_key]
        return x.T.flatten()
    
class reindex_transform():
    def __init__(self, reindex):
        self.reindex = reindex

    def __call__(self, x: torch.Tensor):
        return x[self.reindex]