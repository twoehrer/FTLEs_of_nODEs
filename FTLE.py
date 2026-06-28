#containbs all the relevant functions to compute the finite time lyapunov exponents (FTLEs) used for the plots
#code by Tobias Woehrer


import numpy as np
import torch



from matplotlib.colors import LinearSegmentedColormap
import os

def input_to_output(input, node, time_interval = torch.tensor([0, 10], dtype=torch.float32)):
    return node.flow(input, time_interval)[-1]



def LEs_batch(inputs, node, time_interval = torch.tensor([0, 10], dtype=torch.float32), compute_gradients = False):
    """
    Batched FTLE computation. One batched ODE solve produces every per-sample
    Jacobian via torch.autograd.functional.jacobian, instead of looping over
    inputs as in `LEs`.

    Parameters:
    inputs (torch.Tensor): shape (B, D) batch of input points.
    node: NeuralODE model (must expose `node.flow` accepting a batch).
    time_interval (torch.Tensor): integration window [t0, t1].
    compute_gradients (bool): if True, the result has a graph so it can be used
        inside a training loss.

    Returns:
    torch.Tensor: shape (B, D) — (1/Δt) · log(σ_i) per input. Use .max(dim=1)
    for the maximum FTLE per sample.

    Implementation: the per-sample outputs of `node.flow(batch)` are
    independent, so the full Jacobian J_full of shape (B, D, B, D) is
    block-diagonal. We compute J_full once and extract the (B, D, D) diagonal
    blocks. For B*D up to a few hundred this is much faster than a Python loop
    because the forward + D backward passes run on a single batched tensor.
    """
    t = time_interval[1] - time_interval[0]

    flow_fn = lambda x: node.flow(x, time_interval)[-1]   # (B, D)

    # J_full: (B, D, B, D); only blocks J_full[i, :, i, :] are non-zero.
    J_full = torch.autograd.functional.jacobian(
        flow_fn, inputs, create_graph=compute_gradients
    )

    B = inputs.shape[0]
    idx = torch.arange(B)
    J = J_full[idx, :, idx, :]                            # (B, D, D)

    S = torch.linalg.svdvals(J)                           # (B, D)
    return (1.0 / t) * torch.log(S)



def LE_grid(node, x_amount=100,
            time_interval=torch.tensor([0, 10], dtype=torch.float32),
            chunk_size=256):
    """
    Evaluate FTLEs on an x_amount × x_amount grid over [-2, 2]². Returns
    (output_max, output_min), each shape (x_amount, x_amount).

    Uses `LEs_batch` on chunks of `chunk_size` grid points. Chunking is
    needed because `LEs_batch` materialises a (B, D, B, D) Jacobian; at
    B = x_amount² this would be ~1.5 GB for D = 2 at x_amount = 100.
    """
    x = torch.linspace(-2, 2, x_amount)
    y = torch.linspace(-2, 2, x_amount)
    X, Y = torch.meshgrid(x, y)

    inputs = torch.stack([X, Y], dim=-1).view(-1, 2)

    N = inputs.shape[0]
    max_les = torch.zeros(N)
    min_les = torch.zeros(N)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        les = LEs_batch(inputs[start:end], node, time_interval)   # (B, D), descending
        max_les[start:end] = les[:, 0]
        min_les[start:end] = les[:, -1]

    return max_les.view(x_amount, x_amount), min_les.view(x_amount, x_amount)


def mLEs_fast(inputs, node, time_interval=torch.tensor([0, 10], dtype=torch.float32),
              n_iter=20, compute_gradients=False):
    """
    Power iteration estimate of the max FTLE (largest singular value of the flow Jacobian).

    Uses one JVP + one VJP per iteration instead of materialising the full Jacobian,
    so memory is O(D) per step instead of O(D²). Supports batched inputs.

    Because flow_fn: (B, D) -> (B, D) is block-diagonal (output[i] depends only on
    input[i]), a (B, D) tangent matrix passes through JVP/VJP row-wise, so all B
    power iterations run in a single batched call per step.

    Iterations run without a gradient graph; only the final VJP is differentiated when
    compute_gradients=True (spectral-norm convention: gradient flows only through the
    last step, not through the iteration history).

    Parameters:
        inputs: (B, D) batch of input points
        node: model exposing node.flow(x, time_interval)[-1] -> (B, D)
        n_iter: number of power iterations (20 is usually enough for a clear gap)
        compute_gradients: if True, final result has a grad_fn

    Returns:
        (B,) tensor of max FTLEs  (1/Δt) * log(σ₁)
    """
    t = time_interval[1] - time_interval[0]
    B, D = inputs.shape
    flow_fn = lambda x: node.flow(x, time_interval)[-1]

    v = torch.randn(B, D, device=inputs.device, dtype=inputs.dtype)
    v = v / v.norm(dim=1, keepdim=True)

    # Detached base point — iterations have no connection to inputs' grad_fn
    x_det = inputs.detach().requires_grad_(True)
    u = None
    for _ in range(n_iter):
        _, u = torch.autograd.functional.jvp(flow_fn, x_det, v)   # u = J @ v, (B, D)
        u = (u / u.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
        _, v = torch.autograd.functional.vjp(flow_fn, x_det, u)   # v = J^T @ u, (B, D)
        v = (v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()

    # σ₁ = u^T J v = (J^T u) · v — only this step is differentiated
    _, jT_u = torch.autograd.functional.vjp(
        flow_fn, inputs, u, create_graph=compute_gradients
    )
    sigma_final = (jT_u * v).sum(dim=1)                            # (B,)
    return (1.0 / t) * torch.log(sigma_final)

def mLE_fast_grid(node, x_amount=100,
            time_interval=torch.tensor([0, 10], dtype=torch.float32),
            chunk_size=256):
    """
    Evaluate FTLEs on an x_amount × x_amount grid over [-2, 2]². Returns
    (output_max, output_min), each shape (x_amount, x_amount).

    Uses `LEs_batch` on chunks of `chunk_size` grid points. Chunking is
    needed because `LEs_batch` materialises a (B, D, B, D) Jacobian; at
    B = x_amount² this would be ~1.5 GB for D = 2 at x_amount = 100.
    """
    x = torch.linspace(-2, 2, x_amount)
    y = torch.linspace(-2, 2, x_amount)
    X, Y = torch.meshgrid(x, y)

    inputs = torch.stack([X, Y], dim=-1).view(-1, 2)

    N = inputs.shape[0]
    max_les = torch.zeros(N)
    min_les = torch.zeros(N)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        les = mLEs_fast(inputs[start:end], node, time_interval)   # (B, D), descending
        max_les[start:end] = les[:]

    return max_les.view(x_amount, x_amount)

#not used anymore
def LEs(input, node, time_interval = torch.tensor([0, 10], dtype=torch.float32), compute_gradients = False):
    """
    Compute the Finite-Time Lyapunov Exponents (FTLEs) for a given single input and neural ODE.

    Parameters:
    input (torch.Tensor): The input tensor for which to compute the FTLEs.
    time_interval (torch.Tensor): The time interval over which to compute the FTLEs.
    node (NeuralODE): The neural ODE model.
    compute_gradients (bool): If True, computes the gradients. Also compute the singular vectors (u, v). This is required for training. Otherwise we cannot compute gradients of the FTLEs.
    
    
    Only supports single inputs, no batch inputs. (This is because of the jacobian function)
    """
    
    t = time_interval[1]-time_interval[0]
    
    #fix the node so it is just a input to output of the other variable
    input_to_output_lambda = lambda input: input_to_output(input, node, time_interval)
    
    # Compute the Jacobian matrix
    J = torch.autograd.functional.jacobian(input_to_output_lambda, input, create_graph = compute_gradients)
    
    # Perform Singular Value Decomposition
    _, S, _ = torch.svd(J, compute_uv = compute_gradients)
    
    # Return the maximum singular value
    return 1/t * torch.log(S)