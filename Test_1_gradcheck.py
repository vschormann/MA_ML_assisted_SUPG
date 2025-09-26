from sp_problems import ex1 as sd
from torch_classes.supg_torch import supg_loss
import torch

dtype = torch.double
device = torch.device('cpu')


def test_loss(weights):
    return supg_loss(sd, weights)


sd.set_weights(1e-1)
weights = torch.tensor(sd.yh.x.array, requires_grad=True, dtype=dtype, device=device).view(1, -1)
test = torch.rand_like(weights, dtype=dtype, device=device, requires_grad=True).clip(min=torch.Tensor([0]), max=torch.Tensor([1]))

print(str(torch.autograd.gradcheck(test_loss, (test), eps=1e-4, atol=1e-2, raise_exception=True)))
