#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobias woehrer
"""
##------------#
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from matplotlib.colors import to_rgb
import imageio

from matplotlib.colors import LinearSegmentedColormap
import os

import sys

# Add the parent directory of plots.py to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_gif_shrinkingintervals(model, le_density = 30, point_density = 20, filename = 'LE_shrinking'):
    """
    Creates a GIF showing the evolution of input points and the future Lyapunov exponents. As the time progresses, the remaining Lyapunov exponent intervals shrink, reflecting the model's dynamics.
    
    Parameters:
    - model: The neural ODE model.
    - le_amount: Number of Lyapunov exponent intervals.
    - point_density: Number of vector field points.
    - filename: Base name for the output files.
    """
    import io
    from FTLE import LE_grid

    ###point plot preparation####
    # Define the grid for vector field visualization
    x = torch.linspace(-2,2,point_density)
    y = torch.linspace(-2,2,point_density)
    X, Y = torch.meshgrid(x, y)
    inputs_grid = torch.stack([X,Y], dim=-1)
    inputs_grid = inputs_grid.view(-1,2) #to be able to input all the grid values into the model at once

    #compute traj and colors of grid inputs first, later no more evaluations of model needed
    trajs, colors = input_to_traj_and_color(model, inputs_grid)

    
    T = model.T
    step_size = T/model.time_steps #time step for the integration
    eps = 0.1 * step_size #this should make sure we stay inside an interval with constant parameters for the integration
    t_values = np.arange(0 + eps, T-step_size, step_size) #all discretization steps computed in the nODE flow
    
    images = []
    
    for t in t_values:
    
        ###FTLE plot
        le_interval = torch.tensor([t, T], dtype=torch.float32)

        output_max, _ = LE_grid(model, le_density, le_interval)
        plt.imshow(np.rot90(output_max), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=10)  # Adjust tick label size
        
        ### Plot points from trajectory
        plot_points_from_traj(trajs, colors, model, t) #this + step_size here does not make sense yet must still be explained.
        
        plt.gca().set_aspect('equal', adjustable='box')  # more robust than plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel(r"$x_1$", fontsize=5)
        plt.ylabel(r"$x_2$", fontsize=5)
        plt.tick_params(axis='both', which='major', labelsize=5)
        plt.title(f'Time {t:.2f}, FTLE interval [{le_interval[0].item():.2f}, {le_interval[1].item():.2f}] ', fontsize = 7)
        
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', dpi=200, format='png', facecolor='white')
        buf.seek(0)  # rewind to beginning of buffer
        images.append(imageio.imread(buf))  # or use PIL.Image.open(buf)
        buf.close()
        plt.close()
        print(f'Saved plot for t={t:.1f}')
    
    imageio.mimsave(filename + '.gif', images, fps=4)




def create_gif_subintervals(model, le_density = 30, vf_density = 20, filename = 'LE_per_param'):
    """
    Creates a GIF showing the finite-time Lyapunov exponents for each parameter subinterval.
    
    Parameters:
    - model: The neural ODE model.
    - le_amount: Number of Lyapunov exponent intervals.
    - vf_amount: Number of vector field points.
    - filename: Base name for the output files.
    """
    import io
    from FTLE import LE_grid


    ###point plot preparation####
    # Define the grid for vector field visualization
    x = torch.linspace(-2,2,vf_density)
    y = torch.linspace(-2,2,vf_density)
    X, Y = torch.meshgrid(x, y)
    inputs_grid = torch.stack([X,Y], dim=-1)
    inputs_grid = inputs_grid.view(-1,2) #to be able to input all the grid values into the model at once

    #compute traj and colors of grid inputs first, later no more evaluations of model needed
    trajs, colors = input_to_traj_and_color(model, inputs_grid)

    
    T = model.T
    step_size = T/model.time_steps #time step for the integration
    param_subinterval_len = model.T/model.num_params #time interval has length of the time_steps and the subinterval of constant param has the same length
    eps = 0.1 * step_size #this should make sure we stay inside an interval with constant parameters for the integration
    t_values = np.arange(0 + eps, T-step_size, step_size) #all discretization steps computed in the nODE flow
    
    images = []
    k_running = -1
    for t in t_values:
        
        ###FTLE plot
        k = int(t // param_subinterval_len) #which parameter interval we are in
        le_interval = torch.tensor([0, param_subinterval_len], dtype=torch.float32) + k * param_subinterval_len #the time interval for the FTLE computation
        if k != k_running: #only recompute the FTLE if the parameter interval changed
            k_running = k
            print(f'k = {k}, t = {t:.2f}')
            output_max, _ = LE_grid(model, le_density, le_interval)
            
        plt.imshow(np.rot90(output_max), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=10)  # Adjust tick label size
        
        ### Plot vector field and points from trajectory
        plot_vectorfield(model, t , inputs_grid)
        plot_points_from_traj(trajs, colors, model, t + step_size) #this + step_size here does not make sense yet must still be explained.
        
        plt.gca().set_aspect('equal', adjustable='box')  # more robust than plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel(r"$x_1$", fontsize=5)
        plt.ylabel(r"$x_2$", fontsize=5)
        plt.tick_params(axis='both', which='major', labelsize=5)
        plt.title(f'Time {t:.2f}, FTLE interval [{le_interval[0].item():.2f}, {le_interval[1].item():.2f}] ', fontsize = 7)
        
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', dpi=200, format='png', facecolor='white')
        buf.seek(0)  # rewind to beginning of buffer
        images.append(imageio.imread(buf))  # or use PIL.Image.open(buf)
        buf.close()
        plt.close()
        print(f'Saved plot for t={t:.1f}')
    
    imageio.mimsave(filename + '.gif', images, fps=4)

def input_to_traj_and_color(model, inputs):
    """
    Computes the trajectory of the inputs through the model and colors them based on the model's predictions. Only requires one input to output call of the model.
    """
    model_preds, _ = model(inputs)
    if model.output_dim == 1:
                model_preds = model_preds.squeeze(-1)                   # (N,)
                model_preds = torch.clamp((model_preds + 1) / 2, 0.0, 1.0)
    else:
        m = nn.Softmax()
        model_preds = m(model_preds)
        model_preds = model_preds[:,0]
    model_preds = (model_preds > 0.5).long()

    # colors = ['C1' if model_preds[i] == 1 else 'C0' for i in range(len(model_preds))]
    
    colors = ['#ff7f0e' if model_preds[i] == 0 else '#1f77b4' for i in range(len(model_preds))]

    traj = model.traj.detach().numpy() #the trajectories from the last model call is stored in the anode.
    
    return traj, colors

def plot_points_from_traj(trajs, colors, model, time):
    """
    based on trajectories and corresponding pred color of the model, this function plots the points at a given time.
    model is only needed for the time step and the number of time steps.
    """
    
    step_size = model.T/ model.time_steps #the step size is the time interval divided by the number of time steps.
    index = int(time // step_size) #get the index of the time that is requested. if it is not an integer, it will return the floor value. This can lead to rounding errors. An easy fix can be to add a small epislon to the time value.
    plt.scatter(trajs[index, :, 0], trajs[index, :, 1], marker='o', s = 10, color = colors, linewidth=0.65, edgecolors='black', alpha = 1)

    
    

def plot_points(model, inputs, targets_dummy, time_interval = None, dpi=200, alpha=0.9,
                    alpha_line=0.9, x_lim = [-2,2], y_lim = [-2,2]):
    #targets_dummy is a dummy variable to have the unchanged input for old files. should be removed once the update works.
    from matplotlib import rc
    rc("text", usetex=False)
    font = {'size': 18}
    rc('font', **font)
    
    
    preds, _ = model(inputs)
    m = nn.Softmax() #needs to be modified if used with model that already has a softmax layer
    preds = m(preds)
    preds = preds[:,0]
    targets = (preds > 0.5).long()


    # Define color based on targets

    color = ['C1' if targets[i] == 1 else 'C0' for i in range(len(targets))]
        
    
    
    if time_interval is None:
        time_interval = torch.tensor([0, model.T],dtype=torch.float32)
        
    
    trajectories = model.flow(inputs, time_interval).detach() #output is of dimension [time_steps, number of inputs, dimension per input]

    
    for i in range(inputs.shape[0]):
        plt.scatter(trajectories[-1,i, 0], trajectories[-1,i, 1], marker='o', s = 10, color = color[i],linewidth=0.65, edgecolors='black', alpha = alpha)
        
    
    x_min, x_max = x_lim[0], x_lim[1]
    y_min, y_max = y_lim[0], y_lim[1]
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

def plot_vectorfield(anode, t, inputs, show = False):
    
    
    vectorfield_grid = anode.flow.dynamics(t, inputs)
    x_inputs = inputs[..., 0].detach().numpy()
    y_inputs = inputs[..., 1].detach().numpy()

    x_vf = vectorfield_grid[:, 0].detach().numpy()
    y_vf = vectorfield_grid[:, 1].detach().numpy()
    plt.quiver(x_inputs,y_inputs, x_vf, y_vf, color = 'black', width=0.002, alpha = 0.5)#, alpha = 0.8, headlength=1, headaxislength=2, scale=25, width=0.003)
    if show:
        # plt.xlim(-2, 2)
        # plt.ylim(-2, 2)
        # plt.gca().set_aspect('equal', adjustable='box')  # more robust than plt.axis('equal')
        plt.show()


def plot_trajectory(model, inputs, targets, stepsize, time_interval = None, dpi=200, alpha=0.9,
                    alpha_line=0.9, x_lim = [-2,2], y_lim = [-2,2], show = True):
    """
    This function plots the trajectories of the inputs in the model's flow field. It is used as a visusalizetion tool for the model's behavior to make sure the subintervals and stepsizes are as expected to generate the gifs.
    """
    from matplotlib import rc
    rc("text", usetex=False)
    font = {'size': 18}
    rc('font', **font)

    # Define color based on targets

    color = ['C1' if targets[i,1] > 0.0 else 'C0' for i in range(len(targets))]
        
    
    
    if time_interval is None:
        time_interval = torch.tensor([0, model.T],dtype=torch.float32)
        
    start_time = time_interval[0].item()
    end_time = time_interval[1].item()
    num_steps_interval = int((end_time - start_time) / stepsize)
    # print('amount steps', num_steps_interval)
    integration_time = torch.arange(start_time, end_time + stepsize/100, stepsize) #using end_time + stepsize gave a weird inconsistency between including and excluding the step_size
    # print(integration_time)
    trajectories = model.flow(inputs, integration_time).detach() #output is of dimension [time_steps, number of inputs, dimension per input]

    
    for i in range(inputs.shape[0]):
        plt.plot(trajectories[:,i, 0], trajectories[:,i, 1], linestyle='-', marker='', color = color[i], alpha = alpha_line, linewidth = 0.5)
        
    
    x_min, x_max = x_lim[0], x_lim[1]
    y_min, y_max = y_lim[0], y_lim[1]
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    
    
    if show:
        plt.show()


@torch.no_grad()
def visualize_classification(model, data, label, grad = None, fig_name=None, footnote=None, contour = True, x1lims = [-2, 2], x2lims = [-2, 2]):
    
    
    x1lower, x1upper = x1lims
    x2lower, x2upper = x2lims

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0", zorder = 1)
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1", zorder = 1)

    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)
    # plt.legend()
    if not grad == None:
        for i in range(len(data[:, 0])):
            plt.arrow(data[i, 0], data[i, 1], grad[i, 0], grad[i, 1],
                    head_width=0.05, head_length=0.1, fc='k', ec='k', alpha=0.5, length_includes_head = True)

   
    model.to(device)
    # creates the RGB values of the two scatter plot colors.
    # c0 = torch.Tensor(to_rgba("C0")).to(device)
    # c1 = torch.Tensor(to_rgba("C1")).to(device)

    

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds, _ = model(model_inputs)
    # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    m = nn.Softmax(dim=2)
    # softmax normalizes the model predictions to probabilities
    preds = m(preds)

    # now we only want to have the probability for being in class1 (as prob for class2 is then 1- class1)
    preds = preds[:, :, 0]
    preds = preds.unsqueeze(2)  # adds a tensor dimension at position 2
    # Specifying "None" in a dimension creates a new one. The rgb values hence get rescaled according to the prediction
    # output_image = (1 - preds) * c1[None, None] + preds * c0[None, None]
    # # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    # output_image = output_image.cpu().numpy()
    # plt.imshow(output_image, origin='lower', extent=(x1lower, x1upper, x2lower, x2upper), zorder = -1)
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])
    # plt.axis('scaled')

    # labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.numpy()]
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is black, last is red
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
        z = np.array(preds).reshape(xx1.shape)
        
        levels = np.linspace(0.,1.,8).tolist()
        
        cont = plt.contourf(xx1, xx2, z, levels, alpha=1, cmap=cm, zorder = 0, extent=(x1lower, x1upper, x2lower, x2upper)) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')



    # preds_contour = preds.view(len(x1), len(x1)).detach()
    # plt.contourf(xx1, xx2, preds_contour, alpha=1)
    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
    return fig


@torch.no_grad()
def classification_levelsets(model, fig_name=None, footnote=None, contour = True, plotlim = [-2, 2]):
    
    
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fig = plt.figure(figsize=(5, 5), dpi=100)
    
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.figtext(0.5, -0.02, footnote, ha="center", fontsize=9)

    
   
    model.to(device)

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2  = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy should emulate indexing='xy'
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    
    preds, _ = model(model_inputs)
    print('shape preds', preds.shape)
    if model.output_dim == 1 and model.cross_entropy == False:
        Z = preds.squeeze()  # normalizes the model predictions to probabilities
        Z = torch.clamp((Z + 1)/2, 0.0, 1.0) #normalizes the output trained with MSE to probabilities between 0 and 1
        print('shape Z', Z.shape)
    else:
        Z = preds.softmax(dim=-1)[..., 0]    # (len_x2, len_x1)
        print('shape Z', Z.shape)

# Convert for matplotlib
    X = xx1.detach().cpu().numpy()
    Y = xx2.detach().cpu().numpy()
    Z = Z.detach().cpu().numpy()
    
    print('shape check:',X.shape == Y.shape == Z.shape)
    
    # # dim = 2 means that it normalizes along the last dimension, i.e. along the two predictions that are the model output
    # m = nn.Softmax(dim=2)
    # # softmax normalizes the model predictions to probabilities
    # preds = m(preds)

    #we only need the probability for being in class1 (as prob for class2 is then 1- class1)
   
    
    plt.grid(False)
    plt.xlim([x1lower, x1upper])
    plt.ylim([x2lower, x2upper])

    ax = plt.gca()
    ax.set_aspect('equal') 
    
    if contour:
        colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is orange, last is blue
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=40)
       
        
        levels = np.linspace(0.,1.,8).tolist()
        
        cont = plt.contourf(X, Y, Z, levels, alpha=1, cmap=cm, zorder = 0) #plt.get_cmap('coolwarm')
        cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('prediction prob.')
        
    

    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
    else: plt.show()
        
def loss_evolution(trainer, epoch, filename = '', figsize = None, footnote = None):
    print(f'{epoch = }')
    fig = plt.figure(dpi = 100, figsize=(figsize))
    labelsize = 10

    #plot whole loss history in semi-transparent
    epoch_scale = range(1,len(trainer.histories['epoch_loss_history']) + 1)
    epoch_scale = list(epoch_scale)
    plt.plot(epoch_scale,trainer.histories['epoch_loss_history'], 'k', alpha = 0.5 )
    plt.plot(epoch_scale, trainer.histories['epoch_loss_rob_history'], 'C2--', zorder = -1, alpha = 0.5)
    
    if trainer.eps > 0: #if the trainer has a robustness term
        standard_loss_term = [loss - rob for loss, rob in zip(trainer.histories['epoch_loss_history'],trainer.histories['epoch_loss_rob_history'])]
        plt.plot(epoch_scale, standard_loss_term,'C1--', alpha = 0.5)
        leg = plt.legend(['total loss', 'gradient term', 'standard term'], prop= {'size': labelsize})
    else: leg = plt.legend(['standard loss', '(inactive) gradient term'], prop= {'size': labelsize})
        
    #set alpha to 1
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    plt.plot(epoch_scale[0:epoch], trainer.histories['epoch_loss_history'][0:epoch], color = 'k')
    plt.scatter(epoch, trainer.histories['epoch_loss_history'][epoch-1], color = 'k' , zorder = 1)
    
    plt.plot(epoch_scale[0:epoch], trainer.histories['epoch_loss_rob_history'][0:epoch], 'C2--')
    plt.scatter(epoch, trainer.histories['epoch_loss_rob_history'][epoch - 1], color = 'C2', zorder = 1)
    
    if trainer.eps > 0: #if the trainer has a robustness term
        plt.plot(epoch_scale[0:epoch], standard_loss_term[0:epoch],'--', color = 'C1')
        plt.scatter(epoch, standard_loss_term[epoch - 1], color = 'C1', zorder = 1)
        
    plt.xlim(1, len(trainer.histories['epoch_loss_history']))
    # plt.ylim([0,0.75])
    plt.yticks(np.arange(0,1,0.25))
    plt.grid(zorder = -2)
    # plt.tight_layout()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.set_aspect('auto')
    ax.set_axisbelow(True)
    plt.xlabel('Epochs', size = labelsize)
    if trainer.eps > 0:
        plt.ylabel('Loss Robust', size = labelsize)
        
    else:
        plt.ylabel('Loss Standard', size = labelsize)

    if footnote:
        plt.figtext(0.5, -0.005, footnote, ha="center", fontsize=9)

    if not filename == '':
        plt.savefig(filename + '.png', bbox_inches='tight', dpi=100, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
        
    else:
        plt.show()
        print('no filename given')
        

def comparison_plot(filename1, title1, filename2, title2, filename_output, figsize = None, show = False, dpi = 100):
    plt.figure(dpi = dpi, figsize=figsize)
    plt.subplot(121)
    sub1 = imageio.imread(filename1)
    plt.imshow(sub1)
    plt.title(title1)
    plt.axis('off')

    plt.subplot(122)
    sub2 = imageio.imread(filename2)
    plt.imshow(sub2)
    plt.title(title2)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(filename_output, bbox_inches='tight', dpi=dpi, format='png', facecolor = 'white')
    if show: plt.show()
    else:
        plt.gca()
        plt.close()
        
        
def train_to_classifier_imgs(model, trainer, dataloader, subfolder, num_epochs, plotfreq, filename = '', plotlim = [-2, 2]):
    
    if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    fig_name_base = os.path.join(subfolder,'') #os independent file path

    for epoch in range(0,num_epochs,plotfreq):
        trainer.train(dataloader, plotfreq)
        epoch_trained = epoch + plotfreq
        classification_levelsets(model, fig_name = fig_name_base + filename + str(epoch_trained), footnote = f'epoch = {epoch_trained}', plotlim = plotlim)
        print(f'\n Plot {epoch_trained =}')