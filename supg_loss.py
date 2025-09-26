from torch_classes import supg_torch

def supg_loss(sd, weights):
    result, _ = supg_torch.supg_func_torch.apply(weights, sd)
    return result