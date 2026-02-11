#containbs all the relevant functions to compute the finite time lyapunov exponents (FTLEs) used for the plots


import numpy as np
import torch



from matplotlib.colors import LinearSegmentedColormap
import os

def input_to_output(input, node, time_interval = torch.tensor([0, 10], dtype=torch.float32)):
    return node.flow(input, time_interval)[-1]

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

def LE_grid(node, x_amount = 100, time_interval = torch.tensor([0, 10], dtype=torch.float32)):
        
        x = torch.linspace(-2,2,x_amount)
        y = torch.linspace(-2,2,x_amount)
        X, Y = torch.meshgrid(x, y)

        inputs = torch.stack([X,Y], dim=-1)
        inputs = inputs.view(-1,2) #to be able to loop through all the grid values
        inputs_MLE_max = torch.zeros(x_amount * x_amount)
        inputs_MLE_min = torch.zeros(x_amount * x_amount)



        for i, input in enumerate(inputs):
                
                inputs_MLE_max[i] = torch.max(LEs(input, node, time_interval))
                inputs_MLE_min[i] = torch.min(LEs(input, node, time_interval))
        
        
        output_max = inputs_MLE_max.view(x_amount,x_amount)
        output_min = inputs_MLE_min.view(x_amount,x_amount)
        
        return output_max, output_min