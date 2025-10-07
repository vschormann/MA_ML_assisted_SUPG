from supg.sp_problems import dat1 as dat
from torch_classes.supg_torch import supg_loss
import torch
from supg import supg
import numpy as np


sd = supg.data(*dat, False)
dtype = torch.double
device = torch.device('cpu')


def test_loss(weights):
    return supg_loss(sd, weights)

size = sd.yh.x.index_map.size_local

for i in range(10):
    weights = np.random.rand(size)
    sd.set_weights(weights=weights)
    test = torch.tensor(sd.yh.x.array[:size], dtype=dtype, device=device, requires_grad=True).view(1, -1)

    print(str(i) + ': ' + str(torch.autograd.gradcheck(test_loss, (test), eps=1e-4, atol=1e-2, raise_exception=False)))
