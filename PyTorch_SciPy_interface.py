
from scipy.optimize import minimize, Bounds
import torch

class PyTorch_SciPy_interface():
    def __init__(self, nn, model_eval):
        self.nn = nn
        self.model_eval = model_eval


    def _set_params(self, x):
        pointer = 0
        for p in self.nn.parameters():
            p.grad = None
            numel = p.numel()
            p.data = torch.tensor(x[pointer:pointer+numel]).view_as(p)
            pointer += numel


    def _get_params(self):
        return torch.cat([p.view(-1) for p in self.nn.parameters()])



    def _eval_grad(self, x):
        return torch.cat([p.grad.view(-1) for p in self.nn.parameters()]).detach().numpy()
    
    def _eval(self, x):
        self._set_params(x)
        loss = self.model_eval()
        loss.backward()
        return loss.item()
    
    def optimize(self, algorithm='L-BFGS-B', ftol=1e-16, gtol=1e-16, max_iter=10000):
        return minimize(
            fun=self._eval,
            x0=self._get_params().detach().numpy(),
            jac=self._eval_grad,
            method=algorithm,
            callback=lambda intermediate_result: print(f"J: {intermediate_result.fun}"),
            options={'ftol':ftol, 'gtol':gtol, 'maxiter':max_iter}
        )