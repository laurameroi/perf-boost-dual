import torch, os
from scipy.stats import multivariate_normal # TODO: use something compatible with tensors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')  # or 'Qt5Agg', 'MacOSX', depending on your OS

from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_trajectories(
    y, ybar, n_agents, save_folder, text="", save=True, filename='', T=100,
    dots=False, circles=False, axis=False, min_dist=1, f=5,
    obstacle_centers=None, obstacle_covs=None
):
    ybar = ybar.flatten()

    # fig = plt.figure(f)
    fig, ax = plt.subplots(figsize=(f,f))
    # plot obstacles
    if not obstacle_covs is None:
        assert not obstacle_centers is None
        yy, xx = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
        zz = xx * 0
        for center, cov in zip(obstacle_centers, obstacle_covs):
            distr = multivariate_normal(
                cov=torch.diag(cov.flatten()).detach().clone().cpu().numpy(),
                mean=center.detach().clone().cpu().numpy().flatten()
            )
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    zz[i, j] += distr.pdf([xx[i, j], yy[i, j]])
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax = fig.subplots()
        ax.pcolormesh(xx, yy, zz, cmap='Greys', vmin=z_min, vmax=z_max, shading='gouraud')

    ax.set_title(text)
    colors = ['tab:blue', 'tab:orange']
    state_dim = y.shape[-1]
    for i in range(n_agents):
        ax.plot(
            y[:T+1,state_dim*i].detach().cpu(), y[:T+1,state_dim*i+1].detach().cpu(),
            color=colors[i%2], linewidth=1
        )
        ax.plot(
            y[T:,state_dim*i].detach().cpu(), y[T:,state_dim*i+1].detach().cpu(),
            color='k', linewidth=0.1, linestyle='dotted', dashes=(3, 15)
        )
    for i in range(n_agents):
        ax.plot(
            y[0,state_dim*i].detach().cpu(), y[0,state_dim*i+1].detach().cpu(),
            color=colors[i%2], marker='8'
        )
        ax.plot(
            ybar[state_dim*i].detach().cpu(), ybar[state_dim*i+1].detach().cpu(),
            color=colors[i%2], marker='*', markersize=10
        )

    if dots:
        for i in range(n_agents):
            for j in range(T):
                ax.plot(
                    y[j, state_dim*i].detach().cpu(), y[j, state_dim*i+1].detach().cpu(),
                    color=colors[i%2], marker='o'
                )
    if circles:
        for i in range(n_agents):
            r = min_dist/2
            circle = ax.Circle(
                (y[T, state_dim*i].detach().cpu(), y[T, state_dim*i+1].detach().cpu()),
                r, color=colors[i%2], alpha=0.5, zorder=10
            )
            ax.add_patch(circle)
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    if save:
        fig.savefig(
            os.path.join(save_folder,
                filename+'_trajectories.pdf'),
            format='pdf'
        )
        plt.close()
    else:
        plt.show()


def plot_traj_vs_time(n_agents, y, save_folder, t_end=None, u=None, text="", save=True, filename=''):
    
    if t_end is None:
        t_end = y.shape[-2]
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    state_dim = y.shape[-1]
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, y[:,state_dim*i].detach().cpu())
        plt.plot(t, y[:,state_dim*i+1].detach().cpu())
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    if state_dim==4:
        for i in range(n_agents):
            plt.plot(t, y[:,state_dim*i+2].detach().cpu())
            plt.plot(t, y[:,state_dim*i+3].detach().cpu())
        plt.xlabel(r'$t$')
        plt.title(r'$v(t)$')
        plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i].detach().cpu())
            plt.plot(t, u[:, 2*i+1].detach().cpu())
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig(
            os.path.join(
                save_folder,
                filename + text + '_x_u.pdf'
            ),
            format='pdf'
        )
        plt.close()
    else:
        plt.show()
