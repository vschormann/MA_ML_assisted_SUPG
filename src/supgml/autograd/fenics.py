import torch


class FEniCSx_PyTorch_interface(torch.autograd.Function):
    @staticmethod
    def forward(weights, fs):
        w = weights.view(-1).cpu().detach().numpy()
        fs.set_weights(w)
        err = torch.tensor(fs.loss(), dtype=weights.dtype, device=weights.device)
        return err
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        weights, fs = inputs
        ctx.grad = fs.grad().reshape(-1,1)
        ctx.dtype = weights.dtype
        ctx.device = weights.device

    @staticmethod
    def backward(ctx, grad_output):
        grad = torch.tensor(ctx.grad, dtype=ctx.dtype, device=ctx.device)
        return grad_output * grad, None
    

class fem_solver():
    def __init__(self, fs):
        self.fs = fs
        self.autograd_func = FEniCSx_PyTorch_interface.apply
    def __call__(self, weights):
        return self.autograd_func(weights, self.fs)
    
class batched_loss_fn():
    """Evaluate a collection of differentiable FEM solvers by mesh ID."""

    def __init__(self, solvers):
        self.fsl = {
            int(mesh_id): solver if isinstance(solver, fem_solver) else fem_solver(solver)
            for mesh_id, solver in solvers.items()
        }

    @classmethod
    def from_repository(cls, repository, case_numbers, split="train", variant="standard"):
        solvers = {}
        for number in case_numbers:
            solver, graph = repository.load(number, split=split, variant=variant)
            solvers[int(graph.mesh_id)] = solver
        return cls(solvers)

    def __call__(self, ptr, idx, y):
        loss_vals = [self.fsl[int(idx[i])](y[ptr[i]:ptr[i+1]] ) for i in range(len(ptr)-1)]
        return torch.stack(loss_vals).sum()


def self_supervised_train(model, loader, optimizer, loss_fn, device):
    #model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        ptr = data.ptr
        idx = data.mesh_id
        out = model(data)
        loss = loss_fn(ptr, idx, out)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
