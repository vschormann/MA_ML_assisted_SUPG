import torch

class supg_func_torch(torch.autograd.Function):
    @staticmethod
    def forward(weights, sd):
        w = weights.cpu().detach().numpy()[0]
        sd.set_weights(w)
        err = torch.tensor(sd.global_loss(), dtype=weights.dtype, device=weights.device)
        return err, sd
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        sd, weights = inputs
        err , sd = output
        ctx.sd = sd
        ctx.device = err.device
        ctx.dtype = err.dtype

    @staticmethod
    def backward(ctx, grad_output, grad_sd):
        return grad_output * torch.tensor(ctx.sd.constrained_grad().reshape(1,-1), dtype=ctx.dtype, device=ctx.device), grad_sd
    

def supg_loss(sd, weights):
    autograd_func = supg_func_torch.apply
    loss, _ = autograd_func(weights, sd)
    return loss

