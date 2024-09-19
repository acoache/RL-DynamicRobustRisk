"""
Main testing script:
    -- Dynamic CVaR
    -- portfolio allocation environment

@date: Sept 2024
@author: Anthony Coache
"""
# numpy
import numpy as np
# scipy
from scipy.stats import gaussian_kde
# plotting
import matplotlib.pyplot as plt
# pytorch
import torch as T
# personal modules
from hyperparams import Hyperparameters
from envs import Environment
from utils import directory
from models import PolicyANN
from risk_measures import RiskMeasure
from agents import LearningAlgo
# misc
from utils import colors, rainbow

if __name__ == '__main__':

    """
    Parameters
    """
    repo_name = "results"  # repo name
        
    rm_type = "CVaR02"
    rm_folders = ['agent-cvar0.2-eps0',
                  'agent-cvar0.2-eps1e-6',
                  'agent-cvar0.2-eps1e-2',
                  'agent-cvar0.2-eps1e-1']
    rm_labels = ['eps0',
                 'eps1e-6',
                 'eps1e-2',
                 'eps1e-1']
    alphas = [0.2, 0.2, 0.2, 0.2]
    epsilons = [0.0, 1e-6, 1e-2, 1e-1]
    params = Hyperparameters(type_rm="CVaR", epsilon=0.001, beta=0.2)
        
    # parameters for the cash-flow evolution figure
    seed = 1234
    Nsims = 50_000
    """
    Instantiation of objects
    """

    directory(repo_name)  # create repo for models
    env = Environment(params.env_params)  # create the environment
    risk_measure = RiskMeasure(params.risk_params)  # create the risk measure

    print('\n*** Name of the repository: ', repo_name, ' ***\n')
    print(params)
    
    # create ANNs
    pi = PolicyANN(
        s_size=2+len(env.S0),
        env=env,
        net_params=params.net_params)
    
    """
    Learning algorithm
    """
    # initialize the actor-critic algorithm
    algo = LearningAlgo(env=env,
                        RM=risk_measure,
                        Q=[], Q_target=[],
                        pi=pi, pi_target=[],
                        F=[], F_target=[],
                        mu=[], mu_target=[],
                        params=params.algo_params)
    
    c_paths = np.zeros((Nsims, len(env.t)-1, len(rm_labels)))
    pis_paths = np.zeros((Nsims, len(env.t)-1, len(env.S0), len(rm_labels)))
    
    # figure parameters
    nrows = 1  # number of rows
    ncols = len(rm_labels)  # number of columns
    plt.rcParams.update({'font.size': 16, 'figure.figsize': (24, 6)})
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots(nrows, ncols, sharey='all', sharex='all')
    grid = plt.GridSpec(nrows, ncols)
    
    for idx_method, method in enumerate(rm_folders):
        print('\n*** Method = ', rm_labels[idx_method], ' ***\n')  # print progress
        
        # load weights of trained models
        checkpoint = T.load(method+'/checkpoint.pt', map_location=algo.device)
        algo.pi.load_state_dict(checkpoint['pi_model'])
        
        with T.no_grad():
            # set seed for simulations
            T.manual_seed(seed)
            np.random.seed(seed)
    
            # generate trajectories from the optimal policy
            trajs = algo.sim_trajs(Nsims=Nsims, choose='best')
            c_paths[:,:,idx_method] = trajs["costs"].detach().numpy()
            pis_paths[:,:,:,idx_method] = trajs["actions"].cpu().numpy()
        
        """
        Figure -- Optimal policy
        """
        axes[idx_method].stackplot(env.t[:-1].cpu().numpy(),
                                   np.mean(pis_paths[:,::-1,:,idx_method], axis=0).transpose(),
                                   labels=env.names_tickers[::-1],
                                   colors=rainbow)
    
    axes[0].set_xlim(0,env.t[-2])
    axes[0].set_ylim(0.0,1.0)
    
    # titles for columns
    columns = []
    for idx in range(ncols):
        columns.append(fig.add_subplot(grid[:,idx], frameon=False))
        columns[idx].set_title(rm_labels[idx] + '\n', fontweight='semibold')
        columns[idx].axis('off')

    # labels for all plots
    xyaxis = fig.add_subplot(grid[:,:], frameon=False)
    xyaxis.set_xlabel(r"$t$")
    xyaxis.set_ylabel(r"$\pi_{t}^{(i)}$")
    xyaxis.tick_params(labelcolor='none', which='both',
                       top=False, bottom=False, left=False, right=False)

    # legend
    handles, labels = axes[idx_method].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.90) 
    plt.savefig(repo_name + '/GBM-pis-' + rm_type + '.pdf',
                transparent=True, bbox_inches='tight')
    plt.clf()
    plt.close()
    
    """
    Figure -- Cash-flow evolution
    """
    rewards_total = -1 * np.sum(c_paths, axis=1)
    reward_grid = np.linspace(np.min(rewards_total), np.max(rewards_total), 100)
    
    # figure parameters
    plt.rcParams.update({'font.size': 16, 'figure.figsize': (12, 6)})
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots()    
    
    # plot the PnL distribution and its KDE
    for idx_method, method in enumerate(rm_folders):
        axes.hist(x=rewards_total[:,idx_method], bins=reward_grid,
                  alpha=0.4, color=colors[idx_method], density=True)
    
    plt.legend(rm_labels)
    axes.set_xlabel("Terminal P&L")
    axes.set_ylabel("Density")
    axes.set_xlim(np.min(reward_grid), np.max(reward_grid))
    
    for idx_method, method in enumerate(rm_folders):
        kde = gaussian_kde(rewards_total[:,idx_method], bw_method='silverman')
        axes.plot(reward_grid, kde(reward_grid),
                  color=colors[idx_method], linewidth=1.5)
        axes.axvline(x=np.quantile(rewards_total[:,idx_method], 0.1),
                     linestyle='dashed', color=colors[idx_method], linewidth=1.0)
        axes.axvline(x=np.quantile(rewards_total[:,idx_method], 0.9),
                     linestyle='dashed', color=colors[idx_method], linewidth=1.0)
    
    plt.savefig(repo_name + '/GBM-pnl-' + rm_type + '.pdf',
                transparent=True, bbox_inches='tight')
    plt.clf()
    plt.close()
    
    print('*** Testing phase completed! ***')