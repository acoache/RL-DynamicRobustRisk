"""
Main training script:
    -- Dynamic CVaR
    -- portfolio allocation environment

@date: Sept 2024
@author: Anthony Coache
"""
# pytorch
import torch as T
# personal modules
from hyperparams import Hyperparameters
from envs import Environment
from utils import directory
from models import QCVaRANN, PolicyANN, CDFANN, MuANN
from risk_measures import RiskMeasure
from agents import LearningAlgo
# misc
from time import time

if __name__ == '__main__':

    """
    Parameters
    """
    params = Hyperparameters(type_rm="CVaR", epsilon=0.0, beta=0.2)
    repo_name = "agent-cvar0.2-eps0"
    
    preload = False  # load pre-trained model prior to the training phase
    plot_progress = 25_000  # number of epochs before plotting the policy
    save_progress = 25_000  # number of epochs before saving ANNs
    clean_progress = 2_000  # number of epochs before cleaning cuda cache
    print_progress = 10_000  # number of epochs before printing time progress
    
    """
    Instantiation of objects
    """
    
    print('\n*** Name of the repository: ', repo_name, ' ***\n')
    print(params)
    
    directory(repo_name)  # create repo for models
    directory(repo_name + '/diagnostics')  # create repo for diagnostic figures

    env = Environment(params.env_params)  # create the environment
    risk_measure = RiskMeasure(params.risk_params)  # create the risk measure

    # create ANNs
    Q_main = QCVaRANN(
        s_size=2+len(env.S0),
        a_size=len(env.S0),
        env=env,
        net_params=params.net_params)
    Q_target = QCVaRANN(
        s_size=2+len(env.S0),
        a_size=len(env.S0),
        env=env,
        net_params=params.net_params)
    pi = PolicyANN(
        s_size=2+len(env.S0),
        env=env,
        net_params=params.net_params)
    pi_target = PolicyANN(
        s_size=2+len(env.S0),
        env=env,
        net_params=params.net_params)
    F = CDFANN(
        s_size=2+len(env.S0),
        a_size=len(env.S0),
        env=env,
        net_params=params.net_params)
    F_target = CDFANN(
        s_size=2+len(env.S0),
        a_size=len(env.S0),
        env=env,
        net_params=params.net_params)
    mu = MuANN(
        s_size=2+len(env.S0),
        a_size=len(env.S0),
        env=env,
        net_params=params.net_params)
    mu_target = MuANN(
        s_size=2+len(env.S0),
        a_size=len(env.S0),
        env=env,
        net_params=params.net_params)
    
    """
    Learning algorithm
    """
    # initialize the actor-critic algorithm
    algo = LearningAlgo(env=env,
                        RM=risk_measure,
                        Q=Q_main, Q_target=Q_target,
                        pi=pi, pi_target=pi_target,
                        F=F, F_target=F_target,
                        mu=mu, mu_target=mu_target,
                        params=params.algo_params)
    
    algo.set_train_mode(train=True)  # set models in training mode
    start_time = time()  # start timer
    
    if preload:
        algo.load_models(repo=repo_name, optim=True)  # load trained models
    else:
        # evaluate Q to have a good approximation
        algo.initial_estimates_noeps(Nsims=params.algo_params["Nsims"],
                                     Nepochs=params.algo_params["Nepochs_init"],
                                     Nminibatch=params.algo_params["batch_F"])
        
    for epoch in range(3*params.algo_params["Nepochs"]):
            
        with T.no_grad():
            # generate mini-batch of trajectories
            trajs = algo.sim_trajs(Nsims=params.algo_params["Nsims"],
                                   choose='random')
        
        # estimate the Q-function
        algo.update_Q_noeps(trajs=trajs,
                            Nepochs=params.algo_params["Nepochs_Q"],
                            Nminibatch=params.algo_params["batch_Q"])
        
        # update the policy
        algo.update_pi_noeps(Nepochs=params.algo_params["Nepochs_pi"],
                             Nminibatch=params.algo_params["batch_pi"])
        
        algo.decay_exploration()  # decay the exploration parameters
        
        algo.update_targets()  # update target networks
        
        # plot current policy
        if epoch % plot_progress == 0 or epoch == params.algo_params["Nepochs"]-1:
            algo.plot_diagnostics(repo=repo_name + '/diagnostics',
                                  seed=params.algo_params["seed"])

        # save progress
        if epoch % save_progress == 0:
            algo.save_models(repo=repo_name + '/diagnostics',
                             timestamp=True)
        
        # clean cuda cache and losses
        if epoch % clean_progress == 0 and epoch != 0:
            T.cuda.empty_cache()
            algo.clean_losses()
            
        # print progress
        if epoch % print_progress == 0 or epoch == params.algo_params["Nepochs"]-1:
            algo.print_diagnostic(epoch, start_time)
            start_time = time()
    
    algo.set_train_mode(train=False)  # set models in evaluation mode
    algo.save_models(repo=repo_name)  # save the neural networks
    
    print('*** Training phase completed! ***')
