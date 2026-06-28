#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tobias woehrer
"""
##------------#
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from matplotlib.colors import to_rgb
import imageio

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

import sys

# Add the parent directory of plots.py to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))




# Function to set Global rcParams for consistent figure styling (does not override local settings)
# Better to call this function in the notebook as some plots are located in different files than plots.py

def set_plot_defaults():
    plt.rcParams.update({
        "figure.figsize": (5, 5),     # square figure
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",

        # --- Fonts (match explicit sizes) ---
        "font.size": 10,               # base
        "axes.labelsize": 16,          # x/y labels
        "axes.titlesize": 14,          # title
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


# ── Adversarial attack utilities ──────────────────────────────────────────────

def grad_loss_inputs(model, X, y, loss_func):
    """Gradient of loss w.r.t. inputs X."""
    X_req = X.detach().requires_grad_(True)
    preds, _ = model(X_req)
    loss = loss_func(preds, y)
    loss.backward()
    return X_req.grad.detach()


def find_attacks(model, X, y, adversarial_budget, attack_type='equi', result='success',
                 cross_entropy=False, margin=0.1, equi_amount=36, return_stats=False, verbose=False):
    """
    Find adversarial perturbations for correctly classified inputs.

    X            : (N, 2)  input points
    y            : (N, 2)  vector labels  (y[:,1] < 0 -> class 0 / blue,
                                            y[:,1] > 0 -> class 1 / orange)
    margin       : confidence threshold, applied in two places:
                     (1) base selection for high-conf count — only points with
                         |p - 0.5| >= margin are eligible;
                     (2) output filter — the attacked prediction must satisfy
                         |p_attacked - 0.5| >= margin.
                   n_correct / n_attacks always use margin=0 (every correctly
                   classified point is attacked); n_correct_conf / n_high_conf
                   apply the margin.  margin=0 leaves all counts identical.
                   Uses p = clamp((preds[:,1]+1)/2, 0, 1).
    equi_amount  : number of equidistant directions to try for attack_type='equi'.
    return_stats : if True, returns a 4th value — a dict with n_correct,
                   n_correct_conf, n_attacks, n_high_conf.
    Returns: base_points, base_labels, grad  (high-confidence subset),
             and optionally a stats dict.
    """
    if cross_entropy:
        loss_func = nn.CrossEntropyLoss()
    else:
        loss_func = nn.MSELoss()

    with torch.no_grad():
        preds, _ = model(X)
    p = torch.clamp((preds[:, 1] + 1) / 2, 0.0, 1.0)
    pred_class = (p < 0.5).long()
    true_class = (y[:, 1] < 0).long()

    # All correctly classified (margin=0) — full attack pool
    correct_all  = (pred_class == true_class)
    # Confidently correct — subset eligible for high-confidence count
    correct_conf = correct_all & ((p - 0.5).abs() >= margin)

    X_corr = X[correct_all]
    y_corr = y[correct_all]
    conf_in_corr = correct_conf[correct_all]   # confidence flag within X_corr

    if verbose:
        print(f'Correctly classified: {correct_all.sum().item()},  '
              f'of which confident (margin={margin}): {correct_conf.sum().item()}')

    if adversarial_budget == 0:
        attack_dir = torch.zeros_like(X_corr)
    else:
        if attack_type == 'fgsm':
            attack_dir = grad_loss_inputs(model, X_corr, y_corr, loss_func)
            attack_dir = adversarial_budget * torch.sign(attack_dir)
        elif attack_type == 'l2':
            attack_dir = grad_loss_inputs(model, X_corr, y_corr, loss_func)
            attack_dir = adversarial_budget * torch.nn.functional.normalize(attack_dir)
        elif attack_type == 'equi':
            # Brute-force search over equi_amount evenly spaced directions on the L2 sphere.
            # For each direction, evaluate the true model loss (no linear approximation).
            # The direction with the highest per-sample MSE loss is kept per input point.
            N = len(X_corr)
            angles = torch.linspace(0, 2 * torch.pi, equi_amount + 1)[:-1]  # (equi_amount,)
            directions = torch.stack([torch.sin(angles), torch.cos(angles)], dim=1)  # (equi_amount, 2)

            best_dirs = torch.zeros_like(X_corr)
            best_scores = torch.full((N,), float('-inf'))

            with torch.no_grad():
                for d in directions:
                    X_attacked = X_corr + adversarial_budget * d   # broadcast over N
                    preds_att, _ = model(X_attacked)
                    scores = ((preds_att - y_corr) ** 2).mean(dim=1)  # per-sample MSE
                    improve = scores > best_scores
                    best_dirs[improve] = adversarial_budget * d
                    best_scores[improve] = scores[improve]

            attack_dir = best_dirs
        elif attack_type == 'autoattack':
            raise NotImplementedError("AutoAttack is not implemented yet.")

    X_attacks = X_corr + attack_dir
    with torch.no_grad():
        preds_attacks, _ = model(X_attacks)

    # Flip detection: hard threshold, no margin — every class change is a successful attack
    pred_class_attacked = (preds_attacks[:, 1] < 0).long()
    true_class_corr     = (y_corr[:, 1]         < 0).long()
    if result == 'success':
        flipped = (pred_class_attacked != true_class_corr)
    else:
        flipped = (pred_class_attacked == true_class_corr)

    n_attacks = flipped.sum().item()
    if verbose:
        print(f'Attacks ({result}): {n_attacks} / {correct_all.sum().item()}')

    # High-confidence: base was confident AND attacked output is confident
    p_attacked = torch.clamp((preds_attacks[:, 1] + 1) / 2, 0.0, 1.0)
    high_conf_mask = flipped & conf_in_corr & ((p_attacked - 0.5).abs() >= margin)
    n_high_conf = high_conf_mask.sum().item()
    if verbose:
        print(f'High-confidence attacks (margin={margin}): {n_high_conf} / {n_attacks}')

    out = (X_corr[high_conf_mask], y_corr[high_conf_mask], attack_dir[high_conf_mask])
    if return_stats:
        stats = {
            'n_correct':      correct_all.sum().item(),
            'n_correct_conf': correct_conf.sum().item(),
            'n_attacks':      n_attacks,
            'n_high_conf':    n_high_conf,
        }
        return (*out, stats)
    return out


def _accuracy_from_tensors(model, X, y, margin=0.0):
    with torch.no_grad():
        preds, _ = model(X)
    p = torch.clamp((preds[:, 1] + 1) / 2, 0.0, 1.0)
    pred_class = (p < 0.5).long()
    true_class = (y[:, 1] < 0).long()
    correct = (pred_class == true_class) & ((p - 0.5).abs() >= margin)
    return 100.0 * correct.float().mean().item()


@torch.no_grad()
def compute_accuracy(model, loader, margin=0.0):
    """Accuracy over a DataLoader (returns 0–1 fraction)."""
    dev = next(model.parameters()).device
    _batches = [(X, y) for X, y in loader]
    X_all = torch.cat([X for X, y in _batches], dim=0).to(dev)
    y_all = torch.cat([y for X, y in _batches], dim=0)
    return _accuracy_from_tensors(model, X_all, y_all, margin=margin) / 100.0


def _print_latex_table(rows, N_test, adversarial_budget, equi_amount, margin):
    caption = (
        f'High-confidence adversarial attack success '
        f'(margin~$={margin}$, budget~$={adversarial_budget}$, '
        f'{equi_amount}~equidirectional directions).'
    )
    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\begin{tabular}{lccrrrrr}',
        r'\toprule',
        r'Model & Acc.\ (\%) & Acc.\ conf.\ (\%) & \multicolumn{2}{c}{L2 attacks} & \multicolumn{2}{c}{Equi attacks} & Rob HC\% \\',
        r'\cmidrule(lr){4-5} \cmidrule(lr){6-7}',
        r'& & & all & high-conf & all & high-conf & \\',
        r'\midrule',
    ]
    for r in rows:
        eq_hc_pct = (100.0 * (r['n_correct_conf'] - r['n_eq']) / r['n_correct_conf']) if r['n_correct_conf'] > 0 else float('nan')
        lines.append(
            f'{r["name"]} & {r["acc"]:.1f} & {r["acc_conf"]:.1f}'
            f' & {r["n_l2_all"]} & {r["n_l2"]}'
            f' & {r["n_eq_all"]} & {r["n_eq"]} & {eq_hc_pct:.1f}\\% \\\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        f'\\caption{{{caption}}}',
        r'\label{tab:attacks}',
        r'\end{table}',
    ]
    print('\n' + '\n'.join(lines))


def evaluate_models(models, names, X, y, adversarial_budget,
                    equi_amount=36, margin=0.1,
                    visualize=False, subfolder=None, plotlim=(-2, 2),
                    print_latex=False):
    """
    Evaluate L2 and equi adversarial attack success across a list of models.

    models    : list of NeuralODEvar models
    names     : list of string labels (same length)
    X, y      : full test set as tensors (N, 2) and (N, 2)
    visualize : if True, saves a decision-boundary-with-attacks plot per model
                (requires subfolder)
    print_latex: if True, prints a copy-pasteable LaTeX booktabs table

    Returns a list of dicts, one per model:
        name          : model label
        acc           : test accuracy (%, margin=0)
        acc_conf      : high-confidence accuracy (%, margin applied)
        n_correct     : correctly classified inputs (margin=0)
        n_correct_conf: confidently correct inputs (margin applied)
        n_l2          : high-confidence L2 attack successes
        n_l2_all      : all L2 attack successes (margin=0 flip detection)
        n_eq          : high-confidence equi attack successes
        n_eq_all      : all equi attack successes (margin=0 flip detection)
    """
    N_test = len(X)
    rows = []

    for model, name in zip(models, names):
        acc      = _accuracy_from_tensors(model, X, y, margin=0.0)
        acc_conf = _accuracy_from_tensors(model, X, y, margin=margin)

        _, _, _, stats_l2 = find_attacks(
            model, X, y, adversarial_budget,
            attack_type='l2', margin=margin, return_stats=True, verbose=False)

        X_b_eq, y_b_eq, g_eq, stats_eq = find_attacks(
            model, X, y, adversarial_budget,
            attack_type='equi', equi_amount=equi_amount, margin=margin, return_stats=True, verbose=False)

        rows.append({
            'name':           name,
            'acc':            acc,
            'acc_conf':       acc_conf,
            'n_correct':      stats_l2['n_correct'],
            'n_correct_conf': stats_l2['n_correct_conf'],
            'n_l2':           stats_l2['n_high_conf'],
            'n_l2_all':       stats_l2['n_attacks'],
            'n_eq':           stats_eq['n_high_conf'],
            'n_eq_all':       stats_eq['n_attacks'],
        })

        if visualize and subfolder is not None:
            classification_levelsets_with_attacks(
                model, X_b_eq, y_b_eq, grad=g_eq,
                fig_name=f'{subfolder}/attacks_equi_{name.replace(" ", "_")}',
                plotlim=list(plotlim), margin=margin
            )

    # plain-text summary
    W = 98
    print('\n' + '=' * W)
    print(f'{"Model":<18}  {"Acc":>7}  {"Acc(conf)":>9}  {"L2 all/hc":>14}  {"Equi all/hc":>14}  {"Rob HC%":>9}')
    print('-' * W)
    for r in rows:
        eq_hc_pct = (100.0 * (r['n_correct_conf'] - r['n_eq']) / r['n_correct_conf']) if r['n_correct_conf'] > 0 else float('nan')
        print(f'{r["name"]:<18}  {r["acc"]:>6.1f}%  {r["acc_conf"]:>8.1f}%'
              f'  {r["n_l2_all"]:>4} / {r["n_l2"]:<4}'
              f'  {r["n_eq_all"]:>4} / {r["n_eq"]:<4}'
              f'  {eq_hc_pct:>8.1f}%')
    print('=' * W)

    if print_latex:
        _print_latex_table(rows, N_test, adversarial_budget, equi_amount, margin)

    return rows


def viz_attack(ax, points, y, grad, add_red_squares=False):
    """
    Overlay adversarial attack visuals on ax:
      - arrows from original points to perturbed positions
      - perturbed points coloured by class
      - optional red square markers on perturbed points

    Colour convention: y[:,1] < 0 -> C0 (blue), y[:,1] > 0 -> C1 (orange).
    """
    pts = points.detach().numpy()
    y_np = y.detach().numpy()
    g = grad.detach().numpy()
    attacks = pts + g

    c0 = y_np[:, 1] < 0   # -> blue  (C0)
    c1 = y_np[:, 1] > 0   # -> orange (C1)

    ax.scatter(attacks[c0, 0], attacks[c0, 1], color='C0', edgecolor='black',
               s=20, linewidths=0.5, zorder=3, alpha=0.8)
    ax.scatter(attacks[c1, 0], attacks[c1, 1], color='C1', edgecolor='black',
               s=20, linewidths=0.5, zorder=3, alpha=0.8)
    for i in range(len(pts)):
        ax.arrow(pts[i, 0], pts[i, 1], g[i, 0], g[i, 1],
                 head_width=0.05, head_length=0.06, fc='k', ec='k',
                 alpha=0.5, length_includes_head=True, zorder=3)
    if add_red_squares:
        ax.scatter(attacks[:, 0], attacks[:, 1], facecolor='none', s=70, marker='s',
                   edgecolor='red', linewidths=0.8, zorder=4)


@torch.no_grad()
def classification_levelsets_with_attacks(model, X_base, y_base, grad=None,
                                           fig_name=None, footnote=None, title=None,
                                           plotlim=[-2, 2], num_levels=8,
                                           margin=None):
    """
    Decision boundary contourf + original data scatter + optional attack overlay.

    X_base, y_base : data points to display (e.g. output of find_attacks)
    grad           : perturbation vectors; if None only the points are shown
    """
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    fig, ax = plt.subplots()
    ax.set_ylabel(r'$x_2$')
    ax.set_xlabel(r'$x_1$')
    ax.set_aspect('equal')
    ax.set_xlim([x1lower, x1upper])
    ax.set_ylim([x2lower, x2upper])

    # decision boundary
    x1 = torch.arange(x1lower, x1upper, step=0.01)
    x2 = torch.arange(x2lower, x2upper, step=0.01)
    xx1, xx2 = torch.meshgrid(x1, x2)
    preds, _ = model(torch.stack([xx1, xx2], dim=-1))
    Z = (1 - torch.clamp((preds[..., 1] + 1) / 2, 0.0, 1.0)).numpy()

    if margin is None:
       cm = LinearSegmentedColormap.from_list(
        'Custom', [to_rgb('C1'), [1, 1, 1], to_rgb('C0')], N=40)
       cont = ax.contourf(xx1.numpy(), xx2.numpy(), Z,
                       np.linspace(0., 1., num_levels).tolist(),
                       alpha=0.6, cmap=cm, zorder=0)
    else:
        cm = LinearSegmentedColormap.from_list(
        'Custom', [to_rgb('C1'), [1, 1, 1], to_rgb('C0')], N=5)
        cont = ax.contourf(xx1.numpy(), xx2.numpy(), Z,
                           levels=[0., (0.5-margin) / 2 , 0.5-margin, 0.5 + margin, (0.5 + margin + 1)/2, 1.],
                           cmap=cm,
                           alpha=0.6, zorder=0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(cont, cax=cax).ax.set_ylabel('prediction prob.')

    # data points — same colour convention as viz_attack
    pts = X_base.numpy()
    y_np = y_base.numpy()
    c0 = y_np[:, 1] < 0
    c1 = y_np[:, 1] > 0
    ax.scatter(pts[c0, 0], pts[c0, 1], color='C0', edgecolor='black',
               s=15, linewidths=0.5, zorder=2, alpha=0.8)
    ax.scatter(pts[c1, 0], pts[c1, 1], color='C1', edgecolor='black',
               s=15, linewidths=0.5, zorder=2, alpha=0.8)

    if grad is not None:
        viz_attack(ax, X_base, y_base, grad)

    if title:
        ax.set_title(title)

    if footnote:
        fig.text(0.5, -0.02, footnote, ha='center', fontsize=6)

    if fig_name:
        plt.savefig(fig_name + '.png', dpi=300, format='png', facecolor='white')
        plt.clf()
        plt.close()
    else:
        plt.show()

        



@torch.no_grad()
def trajectory_gif_new(model, inputs, targets, timesteps, dpi=200, alpha=0.9,
                   alpha_line=1, filename='trajectory.gif', axlim = 0, device = 'cpu', fps = 2,
                   colorbar = False):
    
    from matplotlib import rc
    from scipy.interpolate import interp1d
    rc("text", usetex = False)
    font = {'size'   : 18}
    rc('font', **font)
    
    if model.cross_entropy == True:
        raise RuntimeError("This function is not compatible with cross_entropy models at the moment.")

    if not filename.endswith(".gif"):
        raise RuntimeError("Name must end in with .gif, but ends with {}".format(filename))
    base_filename = filename[:-4]


    if model.output_dim == 2:
        color = ['C1' if targets[i,1] > 0.0 else 'C0' for i in range(len(targets))]
    if model.output_dim == 1:
        color = ['C1' if targets[i] < 0.0 else 'C0' for i in range(len(targets))]
        print('passed here')

    trajectories = model.flow.trajectory(inputs, timesteps).detach()
    num_dims = trajectories.shape[2]

    if axlim == 0:        
        x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    else: 
        x_min, x_max = -axlim, axlim  #to normalize for rob and standard nODE
        y_min, y_max = -axlim, axlim   #
        
    if num_dims == 3:
        z_min, z_max = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    if num_dims == 3:
        z_range = z_max - z_min
        z_min -= margin * z_range
        z_max += margin * z_range
        
    T = model.T 
    integration_time = torch.linspace(0.0, T, timesteps)
    
    interp_x = []
    interp_y = []
    interp_z = []
    for i in range(inputs.shape[0]):
        interp_x.append(interp1d(integration_time, trajectories[:, i, 0], kind='cubic', fill_value='extrapolate'))
        interp_y.append(interp1d(integration_time, trajectories[:, i, 1], kind='cubic', fill_value='extrapolate'))
        if num_dims == 3:
            interp_z.append(interp1d(integration_time, trajectories[:, i, 2], kind='cubic', fill_value='extrapolate'))
    
    interp_time = timesteps
    # interp_time = 3 #this was 5 before
    _time = torch.linspace(0., T, interp_time)

    plt.rc('grid', linestyle="dotted", color='lightgray')
    for t in range(interp_time):
        if num_dims == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            label_size = 10
            # plt.rcParams['xtick.labelsize'] = label_size
            # plt.rcParams['ytick.labelsize'] = label_size 
            ax.set_axisbelow(True)
            ax.xaxis.grid(color='lightgray', linestyle='dotted')
            ax.yaxis.grid(color='lightgray', linestyle='dotted')
            ax.set_facecolor('whitesmoke')
            
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            # plt.rc('text', usetex=False)
            # plt.rc('font', family='serif')
            plt.xlabel(r'$x_1$') 
            plt.ylabel(r'$x_2$')
            
            x1 = torch.arange(x_min, x_max, step=0.01, device=device)
            x2 = torch.arange(y_min, y_max, step=0.01, device=device)
            xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
            model_inputs = torch.stack([xx1, xx2], dim=-1)
            
            preds = model.linear_layer(model_inputs)
            

            Z = preds.squeeze()
            if model.output_dim == 2 and model.cross_entropy == False:
                Z = preds[..., 1]
                Z = 1 - torch.clamp((Z + 1)/2, 0.0, 1.0) #normalizes the output trained with MSE to probabilities between 0 and 1
            elif model.output_dim == 1 and model.cross_entropy == False:
                Z = torch.clamp((Z + 1)/2, 0.0, 1.0)
            
            
            plt.grid(False)
            ax = plt.gca()
            ax.set_aspect('equal') 
        
            colors = [to_rgb("C1"), [1, 1, 1], to_rgb("C0")] # first color is orange, last is blue
            cm = LinearSegmentedColormap.from_list(
                "Custom", colors, N=40)
            z = np.array(Z).reshape(xx1.shape)
            
            levels = np.linspace(0.,1.,9).tolist()
            
            cont = plt.contourf(xx1, xx2, z, levels, alpha=0.5, cmap=cm, zorder = 0, extent=(x_min, x_max, y_min, y_max)) #plt.get_cmap('coolwarm')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            if colorbar:
                cbar = fig.colorbar(cont, cax=cax)
                cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                cbar.ax.set_ylabel('prediction prob.')
            else:
                cax.set_axis_off()
            plt.sca(ax)

            plt.scatter([x(_time)[t] for x in interp_x],
                         [y(_time)[t] for y in interp_y], 
                         c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black', zorder=3)

            if t > 0:
                for i in range(inputs.shape[0]):
                    x_traj = interp_x[i](_time)[:t+1]
                    y_traj = interp_y[i](_time)[:t+1]
                    plt.plot(x_traj, y_traj, c=color[i], alpha=alpha_line, linewidth = 0.75, zorder=1)
            
        
        ax.set_aspect('equal')

        plt.savefig(base_filename + "{}.png".format(t),
                    format='png', dpi=dpi, bbox_inches='tight', facecolor = 'white')
        # Save only 3 frames (.pdf for paper)
        # if t in [0, interp_time//5, interp_time//2, interp_time-1]:
        #     plt.savefig(base_filename + "{}.pdf".format(t), format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

    imgs = []
    for i in range(interp_time):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        if i not in [0, interp_time//5, interp_time//2, interp_time-1]: os.remove(img_file)
    imageio.mimwrite(filename, imgs, fps = fps)


@torch.no_grad()
def trajectory_gif_attacks(model, inputs, targets, timesteps, dpi=200, alpha=0.9,
                            alpha_line=1, filename='trajectory_attacks.gif', axlim=0, xrange = None, yrange = None,
                            device='cpu', fps=2, hold_last_frames=8, colorbar = False,
                            title=None, footnote=None):
    """
    Like trajectory_gif_new but marks wrongly-classified endpoints with red squares
    on the final frame, and holds that frame for hold_last_frames extra ticks in the GIF.

    inputs   : (N, 2) — all points to animate (e.g. victim + perturbations)
    targets  : (N, 2) — true labels (y[:,1] < 0 -> blue/C0, y[:,1] > 0 -> orange/C1)
    """
    from scipy.interpolate import interp1d

    if model.cross_entropy:
        raise RuntimeError("Not compatible with cross_entropy models.")
    if not filename.endswith(".gif"):
        raise RuntimeError("filename must end with .gif")
    base_filename = filename[:-4]

    color = ['C1' if targets[i, 1] > 0.0 else 'C0' for i in range(len(targets))]

    trajectories = model.flow.trajectory(inputs, timesteps).detach()
    
    margin = 0.1

    if xrange is None:
        x_min = float(trajectories[:, :, 0].min())
        x_max = float(trajectories[:, :, 0].max())
        x_min -= margin * (x_max - x_min); x_max += margin * (x_max - x_min)
    else:
        x_min, x_max = xrange[0], xrange[1]
    if yrange is None:
        y_min = float(trajectories[:, :, 1].min())
        y_max = float(trajectories[:, :, 1].max())
        y_min -= margin * (y_max - y_min); y_max += margin * (y_max - y_min)
    else:
        y_min, y_max = yrange[0], yrange[1]

    T = model.T
    integration_time = torch.linspace(0.0, T, timesteps)
    interp_x = [interp1d(integration_time, trajectories[:, i, 0], kind='cubic', fill_value='extrapolate')
                for i in range(inputs.shape[0])]
    interp_y = [interp1d(integration_time, trajectories[:, i, 1], kind='cubic', fill_value='extrapolate')
                for i in range(inputs.shape[0])]
    _time = torch.linspace(0., T, timesteps)

    # decision-boundary colormap (shared across frames)
    x1 = torch.arange(x_min, x_max, step=0.01, device=device)
    x2 = torch.arange(y_min, y_max, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds_grid = model.linear_layer(model_inputs)
    Z = preds_grid[..., 1]
    Z = 1 - torch.clamp((Z + 1) / 2, 0.0, 1.0)
    z = np.array(Z).reshape(xx1.shape)
    cm = LinearSegmentedColormap.from_list("Custom", [to_rgb("C1"), [1, 1, 1], to_rgb("C0")], N=40)
    levels = np.linspace(0., 1., 9).tolist()

    # misclassification mask at final positions
    final_pos = trajectories[-1]                        # (N, 2)
    preds_final = model.linear_layer(final_pos)         # (N, 2)
    pred_class = (preds_final[:, 1] < 0).long()
    true_class = (targets[:, 1] < 0).long()
    wrong_mask = (pred_class != true_class)

    plt.rc('grid', linestyle="dotted", color='lightgray')
    for t in range(timesteps):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='lightgray', linestyle='dotted')
        ax.yaxis.grid(color='lightgray', linestyle='dotted')
        ax.set_facecolor('whitesmoke')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.grid(False)
        ax.set_aspect('equal')

        cont = plt.contourf(xx1, xx2, z, levels, alpha=0.5, cmap=cm, zorder=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="5%", pad=0.6)
        if colorbar:
            cb = fig.colorbar(cont, cax=cax)
            cax.yaxis.set_ticks_position('left')
            cax.yaxis.set_label_position('left')
            cb.ax.set_ylabel('prediction prob.')
        else:
            cax.set_axis_off()
        if title:
            ax.set_title(title)
        # plt.get_cmap('coolwarm')

        ax.scatter([fx(_time)[t] for fx in interp_x],
                   [fy(_time)[t] for fy in interp_y],
                   c=color, alpha=alpha, marker='o', linewidth=0.65, edgecolors='black', zorder=3)

        if t > 0:
            for i in range(inputs.shape[0]):
                ax.plot(interp_x[i](_time)[:t+1], interp_y[i](_time)[:t+1],
                        c=color[i], alpha=alpha_line, linewidth=0.75, zorder=1)

        # final frame: red squares on wrongly-classified endpoints
        if t == timesteps - 1:
            final_x = np.array([fx(_time)[t] for fx in interp_x])
            final_y = np.array([fy(_time)[t] for fy in interp_y])
            wrong_np = wrong_mask.numpy()
            if wrong_np.any():
                ax.scatter(final_x[wrong_np], final_y[wrong_np],
                           facecolor='none', s=70, marker='s',
                           edgecolor='red', linewidths=0.8, zorder=4)

        if footnote:
            fig.text(0.5, -0.02, footnote, ha='center', fontsize=6)

        plt.savefig(base_filename + f"{t}.png", format='png', dpi=dpi,
                    bbox_inches='tight', facecolor='white')
        plt.clf()
        plt.close()

    imgs = []
    for i in range(timesteps):
        imgs.append(imageio.imread(base_filename + f"{i}.png"))
    for i in range(timesteps):
        if i not in [0, timesteps // 5, timesteps // 2, timesteps - 1]:
            os.remove(base_filename + f"{i}.png")
    # hold final annotated frame
    for _ in range(hold_last_frames - 1):
        imgs.append(imgs[-1])
    imageio.mimwrite(filename, imgs, fps=fps)


def plot_3d_trajectories(model, inputs, targets, timesteps, attacks=None,
                         attack_targets=None, axlim=0, fig_name=None):
    """
    3-D trajectory plot: x/y = feature space, z = time.

    Trajectories flow from t=0 (bottom) to t=T (top) as coloured 3-D lines.
    The classification decision boundary is embedded as a filled-contour surface
    at z=T so trajectories 'aim' at it like darts.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    with torch.no_grad():
        trajectories = model.flow.trajectory(inputs, timesteps).detach()  # (T, N, 2)

    T = model.T
    time_axis = np.linspace(0.0, T, timesteps)
    N = inputs.shape[0]

    if axlim == 0:
        x_min = float(trajectories[:, :, 0].min()) * 1.1
        x_max = float(trajectories[:, :, 0].max()) * 1.1
        y_min = float(trajectories[:, :, 1].min()) * 1.1
        y_max = float(trajectories[:, :, 1].max()) * 1.1
    else:
        x_min, x_max = -axlim, axlim
        y_min, y_max = -axlim, axlim

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.computed_zorder = False

    # decision boundary surface at z = T
    x1 = torch.arange(x_min, x_max, step=0.05)
    x2 = torch.arange(y_min, y_max, step=0.05)
    xx1, xx2 = torch.meshgrid(x1, x2)
    with torch.no_grad():
        preds_grid = model.linear_layer(torch.stack([xx1, xx2], dim=-1))
    Z_surf = 1 - torch.clamp((preds_grid[..., 1] + 1) / 2, 0.0, 1.0)
    z_surf = Z_surf.numpy()
    cm = LinearSegmentedColormap.from_list("Custom", [to_rgb("C1"), [1, 1, 1], to_rgb("C0")], N=40)
    levels = np.linspace(0., 1., 9).tolist()
    
    # Contour zorder kept at 0
    ax.contourf(xx1.numpy(), xx2.numpy(), z_surf, levels,
                zdir='z', offset=T, cmap=cm, alpha=0.6, zorder=0)

    # trajectories
    for i in range(N):
        traj_x = trajectories[:, i, 0].numpy()
        traj_y = trajectories[:, i, 1].numpy()
        c = 'C1' if targets[i, 1] > 0.0 else 'C0'
        
        # NEW: Added zorder=10 to force the trajectory lines to the absolute foreground
        ax.plot(traj_x, traj_y, time_axis, color=c, linewidth=0.8, alpha=1, zorder=10)
        ax.scatter([traj_x[0]], [traj_y[0]], [time_axis[0]],
                   color=c, s=15, edgecolors='black', linewidths=0.4, zorder=11)

    # final positions with optional red squares for wrong predictions
    final_pos = trajectories[-1]                # (N, 2)
    with torch.no_grad():
        preds_final = model.linear_layer(final_pos)
    pred_class = (preds_final[:, 1] < 0).long()
    true_class = (targets[:, 1] < 0).long()
    wrong_mask = (pred_class != true_class).numpy()

    for i in range(N):
        c = 'C1' if targets[i, 1] > 0.0 else 'C0'
        ax.scatter([float(final_pos[i, 0])], [float(final_pos[i, 1])], [T],
                   color=c, s=25, edgecolors='black', linewidths=0.5, zorder=12) # bumped zorder up
    if wrong_mask.any():
        ax.scatter(final_pos[wrong_mask, 0].numpy(), final_pos[wrong_mask, 1].numpy(),
                   [T] * wrong_mask.sum(),
                   facecolors='none', s=70, marker='s',
                   edgecolors='red', linewidths=0.8, zorder=13) # bumped zorder up

    # Hides the standard Matplotlib box (which also hides standard x/y/z labels)
    ax.set_axis_off()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0, T)

    # remove the grey background panes and grid lines
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)

    # reference axes through the origin
    ax.plot([x_min, x_max], [0, 0], [0, 0], color='black', linewidth=0.6, zorder=0)
    ax.plot([0, 0], [y_min, y_max], [0, 0], color='black', linewidth=0.6, zorder=0)
    ax.plot([0, 0], [0, 0], [0, T], color='black', linewidth=0.6, zorder=0)

    # NEW: Place descriptions manually at the tips of your custom axes
    # The slight offsets (e.g., * 1.05) just prevent the text from overlapping the line end
    ax.text(x_max * 1.05, 0, 0, r'$x_1$', color='black', fontsize=12)
    ax.text(0, y_max * 1.05, 0, r'$x_2$', color='black', fontsize=12)
    ax.text(0, 0, T * 1.05, r'$t$', color='black', fontsize=12)

    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=200, facecolor='white')
        plt.clf()
        plt.close()
    else:
        plt.show()



def _pad_frames_to_common_size(images):
    """Pad GIF frames with white so all share the same shape (imageio requires
    equal-sized frames). With savefig.bbox='tight' the frame sizes vary with the
    colorbar tick labels / title text. Only the in-memory GIF frames are padded;
    saved PNGs keep their tight-bbox sizes so the plot area stays constant.
    Pads right/bottom, so the axes stay anchored at the top-left across frames."""
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    padded = []
    for im in images:
        ph, pw = h - im.shape[0], w - im.shape[1]
        if ph or pw:
            im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode='constant', constant_values=255)
        padded.append(im)
    return padded


def create_gif_growinginterval(model, le_density = 30, vf_density = 20, filename = 'LE_growing', save_pngs = False):
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
    t_values = np.arange(0 + eps, T, step_size) #all discretization steps computed in the nODE flow
    
    images = []
    for i, t in enumerate(t_values):
        
        ###FTLE plot
        le_interval = torch.tensor([0, t], dtype=torch.float32) #the time interval for the FTLE computation
        
        print(f't = {t:.2f}')
        output_max, _ = LE_grid(model, le_density, le_interval)

        fig = plt.figure(figsize=(5, 5))
        im = plt.imshow(np.rot90(output_max), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.sca(ax)

        ### Plot vector field and points from trajectory
        plot_vectorfield(model, t, inputs_grid)
        plot_points_from_traj(trajs, colors, model, t)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.tick_params(axis='both', which='major')
        plt.title(f'Time {t:.2f}, FTLE interval [{le_interval[0].item():.2f}, {le_interval[1].item():.2f}] ')

        buf = io.BytesIO()
        plt.savefig(buf, dpi=200, format='png', facecolor='white', bbox_inches=None)
        if save_pngs:
            plt.savefig(filename + f'_{i}.png', dpi=200, format='png', facecolor='white', bbox_inches=None)
            print(f'Saved PNG for t={t:.1f}, i={i}')
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close()
        print(f'Generated plot for t={t:.1f}')
    
    imageio.mimsave(filename + '.gif', _pad_frames_to_common_size(images), fps=4)

def create_gif_shrinkingintervals(model, le_density = 30, point_density = 20, filename = 'LE_shrinking', save_pngs = False):
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
    
    for i, t in enumerate(t_values):
    
        ###FTLE plot
        le_interval = torch.tensor([t, T], dtype=torch.float32)

        output_max, _ = LE_grid(model, le_density, le_interval)
        plt.figure(figsize=(5, 5))
        im = plt.imshow(np.rot90(output_max), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.sca(ax)

        ### Plot points from trajectory
        plot_points_from_traj(trajs, colors, model, t)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.tick_params(axis='both', which='major')
        plt.title(f'Time {t:.2f}, FTLE interval [{le_interval[0].item():.2f}, {le_interval[1].item():.2f}] ')

        if save_pngs:
            plt.savefig(filename + f'_{i}.png', dpi=200, format='png', facecolor='white', bbox_inches=None)
            print(f'Saved PNG for t={t:.1f}, i={i}')

        buf = io.BytesIO()
        plt.savefig(buf, dpi=200, format='png', facecolor='white', bbox_inches=None)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close()
        print(f'Saved plot for t={t:.1f}')

    imageio.mimsave(filename + '.gif', _pad_frames_to_common_size(images), fps=4)




def create_gif_subintervals(model, le_density = 30, vf_density = 20, filename = 'LE_per_param', save_pngs = False):
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
    t_values = np.arange(0 + eps, T, step_size) #all discretization steps computed in the nODE flow
    
    images = []
    k_running = -1
    for i, t in enumerate(t_values):
        
        ###FTLE plot
        k = int(t // param_subinterval_len) #which parameter interval we are in
        le_interval = torch.tensor([0, param_subinterval_len], dtype=torch.float32) + k * param_subinterval_len #the time interval for the FTLE computation
        if k != k_running: #only recompute the FTLE if the parameter interval changed
            k_running = k
            print(f'k = {k}, t = {t:.2f}')
            output_max, _ = LE_grid(model, le_density, le_interval)
            
        plt.figure(figsize=(5, 5))
        im = plt.imshow(np.rot90(output_max), origin='upper', extent=(-2, 2, -2, 2), cmap='viridis')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.sca(ax)

        ### Plot vector field and points from trajectory
        plot_vectorfield(model, t, inputs_grid)
        plot_points_from_traj(trajs, colors, model, t)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.tick_params(axis='both', which='major')
        plt.title(f'Time {t:.2f}, FTLE interval [{le_interval[0].item():.2f}, {le_interval[1].item():.2f}] ')

        buf = io.BytesIO()
        plt.savefig(buf, dpi=200, format='png', facecolor='white', bbox_inches=None)
        if save_pngs:
            if i == 0 or int((t + step_size) // param_subinterval_len) != k_running or i + 1 == len(t_values):
                plt.savefig(filename + f'_{i}.png', dpi=200, format='png', facecolor='white', bbox_inches=None)
                print(f'Saved PNG for t={t:.1f}, k={k}')
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        plt.close()
        print(f'Generated plot for t={t:.1f}')
    
    imageio.mimsave(filename + '.gif', _pad_frames_to_common_size(images), fps=4)

@torch.no_grad()
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

@torch.no_grad()
def plot_points_from_traj(trajs, colors, model, time):
    """
    based on trajectories and corresponding pred color of the model, this function plots the points at a given time.
    model is only needed for the time step and the number of time steps.
    """
    
    step_size = model.T/ model.time_steps #the step size is the time interval divided by the number of time steps.
    index = int(time // step_size) #get the index of the time that is requested. if it is not an integer, it will return the floor value. This can lead to rounding errors. An easy fix can be to add a small epislon to the time value.
    plt.scatter(trajs[index, :, 0], trajs[index, :, 1], marker='o', s = 10, color = colors, linewidth=0.65, edgecolors='black', alpha = 1)

    
    
@torch.no_grad()
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

class _PhantomMinusFormatter(mticker.ScalarFormatter):
    """ScalarFormatter that pads non-negative labels with a trailing phantom
    minus (figure space + thin space ≈ width of U+2212), so tight-bbox figure
    dimensions don't depend on whether negative ticks are present. The padding
    is appended so the numbers stay flush with the colorbar; the gap opens
    towards the colorbar label instead."""
    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)
        if s and s[0] not in ('−', '-'):
            s = s + '  '
        return s

def plot_FTLEs(LE_values, title = None, vmin=None, vmax=None, filename = None):
    fig, ax = plt.subplots()
    ax_lim = 2
    im = ax.imshow(np.rot90(LE_values), origin='upper', extent=(-ax_lim, ax_lim, -ax_lim, ax_lim), cmap='viridis', vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Lyapunov Exponent')
    cbar.ax.yaxis.set_major_formatter(_PhantomMinusFormatter())

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if filename is not None:
        fig.savefig(filename + '.png')
        plt.close(fig)
    else:
        plt.show()
        
@torch.no_grad()
def classification_levelsets(model, fig_name=None, footnote=None, contour = True, plotlim = [-2, 2], alpha = 1, num_levels = 8):
    
    
    x1lower, x1upper = plotlim
    x2lower, x2upper = plotlim

    device = next(model.parameters()).device

    fig = plt.figure()
    
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")

    
   
    model.to(device)

    x1 = torch.arange(x1lower, x1upper, step=0.01, device=device)
    x2 = torch.arange(x2lower, x2upper, step=0.01, device=device)
    xx1, xx2  = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy should emulate indexing='xy'
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    
    preds, _ = model(model_inputs)
    
    if model.output_dim == 1 and model.cross_entropy == False:
        Z = preds.squeeze() 
        Z = torch.clamp((Z + 1)/2, 0.0, 1.0) #normalizes the output trained with MSE to probabilities between 0 and 1
        print('shape Z', Z.shape)
    elif model.output_dim == 2 and model.cross_entropy == False:
        Z = preds.squeeze() 
        Z = preds[..., 1]
        print('shape preds', preds.shape)
        Z = 1 - torch.clamp((Z + 1)/2, 0.0, 1.0) #normalizes the output trained with MSE to probabilities between 0 and 1
    else:
        Z = preds.softmax(dim=-1)[..., 0]    # (len_x2, len_x1) THIS NEEDS FIXING IF NOT CE! RESCALING

# Convert for matplotlib
    X = xx1.detach().cpu().numpy()
    Y = xx2.detach().cpu().numpy()
    Z = Z.detach().cpu().numpy()
    
    
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

        levels = np.linspace(0., 1., num_levels).tolist()

        cont = ax.contourf(X, Y, Z, levels, alpha=alpha, cmap=cm, zorder=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(cont, cax=cax)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        cbar.ax.set_ylabel('prediction prob.')
        
    
        
    plt.figtext(0.5, -0.02, footnote, ha="center", fontsize = 6)
    

    if fig_name:
        plt.savefig(fig_name + '.png', bbox_inches='tight', dpi=300, format='png', facecolor = 'white')
        plt.clf()
        plt.close()
    else: plt.show()


