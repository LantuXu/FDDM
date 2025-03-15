import torch

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_dtype('cuda'),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):

        # Re-enable gradient calculation for the re-computation of the forward pass
        with torch.enable_grad():
            # Recompute the inputs from the saved context
            detached_inputs = [x.detach().requires_grad_(True) for x in ctx.input_tensors]

            # Apply the same autocast settings as during the forward pass
            with torch.amp.autocast(device_type="cuda", **ctx.gpu_autocast_kwargs):
                # Re-run the forward function with the detached inputs to get new outputs
                output_tensors = ctx.run_function(*detached_inputs)

            # Compute gradients of the output w.r.t. the input tensors
            if isinstance(output_tensors, tuple):
                output_tensors = torch.cat(output_tensors, dim=0)
            grad_input = torch.autograd.grad(
                output_tensors,
                detached_inputs + list(ctx.input_params),
                output_grads,
                allow_unused=True,
                retain_graph=False
            )

        # Return the computed gradients, None for the run_function and length arguments
        return (None, None) + grad_input
def checkpoint(func, inputs, params, flag):

    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)