#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adapted from https://github.com/EmilienDupont/augmented-neural-odes
"""
##------------#
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
# from adjoint_neural_ode import adj_Dynamics

#odeint Returns:
#         y: Tensor, where the first dimension corresponds to different
#             time points. Contains the solved value of y for each desired time point in
#             `t`, with the initial value `y0` being the first element along the first
#             dimension.


MAX_NUM_STEPS = 1000



# Useful dicos:
activations = {'tanh': nn.Tanh(),
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid(),
                'leakyrelu': nn.LeakyReLU(negative_slope=0.25, inplace=True),
}
architectures = {'inside': -1, 'outside': 0, 'bottleneck': 1}




class Dynamics(nn.Module):
    """
    The nonlinear, right hand side $f(u(t), x(t)) of the neural ODE.
    We distinguish the different structures defined in the dictionary "architectures" just above.
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, 
                non_linearity='tanh', architecture='inside', T=10, time_steps=10):
        super(Dynamics, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim

        if non_linearity not in activations.keys() or architecture not in architectures.keys():
            raise ValueError("Activation function or architecture not found. Please reconsider.")
        
        self.non_linearity = activations[non_linearity]
        self.architecture = architectures[architecture]
        self.T = T
        self.time_steps = time_steps
        
        if self.architecture > 0:
            ##-- R^{d_aug} -> R^{d_hid} layer -- 
            blocks1 = [nn.Linear(self.input_dim, hidden_dim) for _ in range(self.time_steps)]
            self.fc1_time = nn.Sequential(*blocks1) 
            ##-- R^{d_hid} -> R^{d_aug} layer --
            blocks3 = [nn.Linear(hidden_dim, self.input_dim) for _ in range(self.time_steps)]
            self.fc3_time = nn.Sequential(*blocks3)
        else:
            ##-- R^{d_hid} -> R^{d_hid} layer --
            blocks = [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.time_steps)]
            self.fc2_time = nn.Sequential(*blocks)
        
    def forward(self, t, x):
        """
        The output of the class -> f(x(t), u(t)).
        f(x(t), u(t)) = f(x,u^k)
        
        """
        dt = self.T/self.time_steps   #here was no -1 before which does not fit with adjoint solver otherwise
        k = int(t/dt - 0.0001) #without the -0.0001 i ran into the case that odeint wants to evaluate at final time T which does not have a k defined. int(-0.0001) = 0 so a negative number is not a problem
        
        if self.architecture < 1:
            w_t = self.fc2_time[k].weight
            b_t = self.fc2_time[k].bias
            if self.architecture < 0:                               # w(t)\sigma(x(t))+b(t)  inner
                out = self.non_linearity(x).matmul(w_t.t()) + b_t        
            else:                                                   # \sigma(w(t)x(t)+b(t))   outer
                out = self.non_linearity(x.matmul(w_t.t())+b_t)
        else:                                                       # w1(t)\sigma(w2(t)x(t)+b2(t))+b1(t) bottle-neck
            w1_t = self.fc1_time[k].weight
            b1_t = self.fc1_time[k].bias
            w2_t = self.fc3_time[k].weight
            b2_t = self.fc3_time[k].bias
            #out = self.non_linearity(x.matmul(w1_t.t()) + b1_t)
            #out = out.matmul(w2_t.t()) + b2_t
            
            #x.matmul(w1_t.t()) is the same as torch.matmul(w1_t,x) simple matrix-vector multiplication

            #Domenec Test
            out1 = torch.sqrt(self.non_linearity(x.matmul(w1_t.t())+torch.ones(self.hidden_dim) + b1_t) + 1e-6*torch.ones(self.hidden_dim))-torch.sqrt(1e-6*torch.ones(self.hidden_dim))
            out2 = torch.sqrt(self.non_linearity(-x.matmul(w1_t.t())+torch.ones(self.hidden_dim) + b1_t) + 1e-6*torch.ones(self.hidden_dim))-torch.sqrt(1e-6*torch.ones(self.hidden_dim))
            out = torch.min(out1, out2)
            out = out.matmul(w2_t.t())
        return out
    
    
class Dynamics_with_layers(nn.Module):
    """
    The nonlinear, right hand side $f(u(t), x(t)) of the neural ODE.
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, 
                non_linearity='tanh', T=10, layers_hidden = 4):
        super(Dynamics_with_layers, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim

        if non_linearity not in activations.keys():
            raise ValueError("Activation function not found. Please reconsider.")
        
        self.non_linearity = activations[non_linearity]
        self.T = T
        self.layers_hidden = layers_hidden
        

        ##-- R^{d_aug} -> R^{d_hid} layer -- 
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        ##-- R^{d_hid} -> R^{d_aug} layer --
        blocks_hidden = [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.layers_hidden)]
        self.fc2 = nn.Sequential(*blocks_hidden)
        self.fc3 = nn.Linear(self.hidden_dim, self.input_dim)

        
    def forward(self, t, x):
        """
        The output of the class -> f(x(t), u(t)).
        f(x(t), u(t)) = f(x,u^k)
        
        """
        out = self.fc1(x)
        out = self.non_linearity(out)
        for k in range(self.layers_hidden):
            out = self.fc2(out)
            out = self.non_linearity(out)
        out = self.fc3(out)
        return out
    



class Semiflow(nn.Module):
    """
    Given the dynamics f, generate the semiflow by solving x'(t) = f(u(t), x(t)).
    We concentrate on the forward Euler method - the user may change this by using
    other methods from torchdiffeq in the modules odeint and odeint_adjoint.

    ***
    - dynamics denotes the instance of the class Dynamics, defining the dynamics f(x,u)
    ***
    """
    def __init__(self, device, dynamics, tol=1e-3, adjoint=False, T=10, time_steps=10):
        super(Semiflow, self).__init__()
        self.adjoint = adjoint 
        self.device = device
        self.dynamics = dynamics
        self.tol = tol
        self.T = T
        self.time_steps = time_steps
        


    def forward(self, x, eval_times=None):
    
        dt = self.T/self.time_steps

        if eval_times is None:
            integration_time = torch.tensor([0, self.T]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.dynamics.augment_dim > 0:
            x = x.view(x.size(0), -1)
            aug = torch.zeros(x.shape[0], self.dynamics.augment_dim).to(self.device)
            x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:  
            out = odeint_adjoint(self.dynamics, x_aug, integration_time, method='euler', options={'step_size': dt})
            
            # out = odeint_adjoint(self.dynamics, x_aug, integration_time, method='dopri5', rtol = 0.1, atol = 0.1)
        else:
            out = odeint(self.dynamics, x_aug, integration_time, method='euler', options={'step_size': dt})
            # out = odeint(self.dynamics, x_aug, integration_time, method='dopri5', rtol = 0.1, atol = 0.1)
            
            
        if eval_times is None:
            return out[1] 
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., self.T, timesteps)
        return self.forward(x, eval_times=integration_time)

class NeuralODEvar(nn.Module):
    """
    Returns the flowmap of the neural ODE, i.e. x\mapsto\Phi_T(x), 
    where \Phi_T(x) might be the solution to the neural ODE, or the
    solution composed with a projection. 
    
    ***
    - output dim is an int the dimension of the labels.
    - architecture is a string designating the structure of the dynamics f(x,u)
    - fixed_projector is a boolean indicating whether the output layer is trained or not
    ***
    """
    def __init__(self, device, data_dim, hidden_dim, output_dim=2,
                 augment_dim=0, non_linearity='tanh',
                 tol=1e-3, adjoint=False, architecture='inside', 
                 T=10, time_steps=10, num_params = 5,
                 cross_entropy=True, fixed_projector=False, layers_hidden = 0):
        super(NeuralODEvar, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        if output_dim == 1 and cross_entropy:
            #output_dim = 1 pour MSE; >=2 pour cross entropy for binary classification.
            raise ValueError('Incompatible output dimension with loss function.')
        self.output_dim = output_dim
        self.tol = tol
        self.T = T
        self.time_steps = time_steps
        self.num_params = num_params #this should describe the amount of piecewise constant parameters exist. i.e. T/num_params
        self.architecture = architecture
        self.cross_entropy = cross_entropy
        self.fixed_projector = fixed_projector
        self.non_linearity_str = non_linearity  # store string for serialization
        self.adjoint = adjoint
        self.layers_hidden = layers_hidden
        if layers_hidden > 0: #if the right hand side has higher dimensional parameters in form of hidden layers (these are not discretization layers but right hand side function layers)
            dynamics = Dynamics_with_layers(device, data_dim, hidden_dim, augment_dim, non_linearity, self.T, layers_hidden = layers_hidden)
            print('hidden layers found')
        else:
            dynamics = Dynamics(device, data_dim, hidden_dim, augment_dim, non_linearity, architecture, self.T, self.num_params)
        
        self.flow = Semiflow(device, dynamics, tol, adjoint, T,  time_steps) #, self.adj_flow
        self.linear_layer = nn.Linear(self.flow.dynamics.input_dim,
                                         self.output_dim)
        self.non_linearity = nn.Tanh() #not really sure why this is here
        self.to(device)
        
    def forward(self, x, return_features=False):
        
        features = self.flow(x)

        self.traj = self.flow.trajectory(x, self.time_steps)
        pred = self.linear_layer(features)
        self.proj_traj = self.linear_layer(self.traj)
        # if not self.cross_entropy:
        #     pred = self.non_linearity(pred)
        #     self.proj_traj = self.non_linearity(self.proj_traj)
        
        if return_features:
            return features, pred
        
        return pred, self.proj_traj
    
    def footnote(self) -> str:
        return (f"num_params={self.num_params}, time steps={self.time_steps}, "
                f"nonlin={self.non_linearity}, cross_e={self.cross_entropy}, "
                f"arch={self.architecture}, "
                f"hidden_dim={self.hidden_dim}, "
                f"output_dim={self.output_dim}, layers_hidden={self.layers_hidden}")
        

