"""
Hyperparameters:
    -- Parameters for the environment, risk measure, and algorithm
    -- Function to print all hyperparameters

@date: Sept 2024
@author: Anthony Coache 
"""
# pytorch
import torch as T
# misc
import psutil


class Hyperparameters():
    def __init__(self, type_rm="mean", epsilon=0.1, beta=None):
        """constructor
        """
        self.env_params = {'r0': 0.0,  # initial interest rate
                           'w0': 1.0,  # initial wealth
                           'T': 1,  # time horizon of the problem
                           'dt': 1/12,  # time elapsed between periods
                           'device': T.device('cuda:0' if T.cuda.is_available() else 'cpu')}  # device

        self.risk_params = {"type": type_rm,  # type of the risk measure
                            "beta": beta,  # beta of the alpha-beta risk
                            "epsilon": epsilon,  # tolerance for Wasserstein uncertainty
                            'device': T.device('cuda:0' if T.cuda.is_available() else 'cpu')}  # device
        
        self.net_params = {'layers_pi': 5,  # number of layers in pi
                           'layers_Q': 6,  # number of layers in Q
                           'layers_F': 7,  # number of layers in F
                           'hidden_pi': 32,  # number of hidden nodes in pi
                           'hidden_Q': 32,  # number of hidden nodes in Q
                           'hidden_F': 32,  # number of hidden nodes in F
                           'lr_pi': 3e-6,  # learning rate of pi
                           'lr_Q': 5e-4,  # learning rate of Q
                           'lr_F': 5e-4,  # learning rate of F
                           'gamma_pi': 0.999997,  # scheduler rate of pi
                           'gamma_Q': 0.999997,  # scheduler rate of Q
                           'gamma_F': 0.999997,  # scheduler rate of F
                           'device': T.device('cuda:0' if T.cuda.is_available() else 'cpu')}  # device
        
        self.algo_params = {'Nepochs': 300_000,  # number of epochs
                            'Nepochs_init': 5,  # number of inner epochs initially
                            'Nepochs_F': 5,  # number of inner epochs for F
                            'Nepochs_Q': 5,  # number of inner epochs for Q
                            'Nepochs_mu': 5,  # number of inner epochs for mu
                            'Nepochs_pi': 1,  # number of inner epochs for pi
                            'Nsims': 2_048,  # number of full trajectories
                            'batch_F': 128,  # mini-batch size for F
                            'batch_Q': 128,  # mini-batch size for Q
                            'batch_mu': 128,  # mini-batch size for mu
                            'batch_pi': 128,  # mini-batch size for pi
                            'tau_target': 0.008,  # rate to update target networks
                            'p_ex': 0.5,  # exploratory noise probability
                            'min_p_ex': 0.5,  # minimal exploration probability
                            'len_z_grid': 501,  # size of the z partition for adversary
                            'Ncpus': int(psutil.cpu_count(logical=False)),  # number of CPUs
                            'seed': 1234}  # set seed for replication purposes
    
    def __str__(self):
        descrp = '*** Parameters for the environment ***\n'
        for key, val in self.env_params.items():
            descrp += '* {}: {}\n'.format(key, val)
        descrp += '\n*** Parameters for the risk measure ***\n'
        for key, val in self.risk_params.items():
            descrp += '* {}: {}\n'.format(key, val)
        descrp += '\n*** Parameters for the ANNs ***\n'
        for key, val in self.net_params.items():
            descrp += '* {}: {}\n'.format(key, val)
        descrp += '\n*** Parameters for the algorithm ***\n'
        for key, val in self.algo_params.items():
            descrp += '* {}: {}\n'.format(key, val)
        return descrp + '\n'
