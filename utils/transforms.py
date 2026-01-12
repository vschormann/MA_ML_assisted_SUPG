import torch


def create_cell_ind_to_grid(H, W):
    w, h = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing="ij"
    )

    d = (H - 1 - w) + h          # diagonal index
    order = torch.argsort(
        d.flatten() * max(H, W) + w.flatten()
    )

    out = torch.empty(H * W, dtype=torch.long)
    out[order] = torch.arange(H * W)

    return out.view(H, W)

def create_flattening_index_set(H,W, continuous_traversal=False):
    h, w = torch.meshgrid(
        torch.arange(H),
        torch.arange(W),
        indexing="ij"
    )
    if continuous_traversal:
        diag_mask = h+w
        diag_mask = torch.ones_like(diag_mask)-diag_mask%2

        sgn_fn = torch.where(diag_mask%2==1, 1, -1)

        diag_key = ((1+sgn_fn/(2*H))*h + w).flip(0)
    else:
        diag_key = (h + (1+1/(2*W))*(w)).flip(0)

    return torch.argsort(diag_key.flatten())

def cells_from_flat_array_ind(H,W):
    ctg = create_cell_ind_to_grid(H,W)
    bctg = torch.concat((ctg[0,:],ctg[-1,:],ctg[:,0],ctg[:,-1])).unique()
    combined = torch.cat((bctg, torch.arange(H*W)))

    uniques, counts = combined.unique(return_counts=True)
    difference = uniques[counts == 1]
    return difference

class flattening_transform:
    def __init__(self, flat_key: torch.Tensor, C: int, pseudo_channel=None):
        self.flat_key = flat_key
        self.C = C
        self.pseudo_channel = pseudo_channel

    def __call__(self, x: torch.Tensor):
        x = x.reshape(self.C,-1)[:,self.flat_key]
        if self.pseudo_channel:
            return x.T.flatten().reshape(1,-1)
        return x.T.flatten()
    
class channeled_flattening_transform:
    def __init__(self, flat_key: torch.Tensor, C: int):
        self.flat_key = flat_key
        self.C = C

    def __call__(self, x: torch.Tensor):
        x = x.reshape(self.C,-1)[:,self.flat_key]
        return x
    
class reindex_transform():
    def __init__(self, reindex):
        self.reindex = reindex

    def __call__(self, x: torch.Tensor):
        return x[self.reindex]
    
class reshape_transform:
    def __init__(self, H:int, W:int, C:int=1):
        self.idx = create_cell_ind_to_grid(H,W).flatten()
        self.H = H
        self.W = W
        self.C = C

    def __call__(self, x: torch.Tensor):
        return x[self.idx].reshape(self.C,self.H, self.W)