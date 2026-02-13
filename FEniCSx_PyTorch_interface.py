import torch



class FEniCSx_PyTorch_interface(torch.autograd.Function):
    @staticmethod
    def forward(weights, sd):
        w = weights.cpu().detach().numpy()
        sd.set_weights(w)
        err = torch.tensor(sd.loss(), dtype=weights.dtype, device=weights.device)
        return err
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        weights, sd = inputs
        ctx.grad = sd.grad().reshape(1,-1)
        ctx.dtype = weights.dtype
        ctx.device = weights.device

    @staticmethod
    def backward(ctx, grad_output):
        grad = torch.tensor(ctx.grad, dtype=ctx.dtype, device=ctx.device)
        return grad_output * grad, None
    
