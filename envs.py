"""
Environment(s):
    -- Portfolio allocation with different price dynamics
    -- Saved dynamics for Github

@date: Sept 2024
@author: Anthony Coache
"""
# numpy
import numpy as np
# pytorch
import torch as T


class Environment():
    def __init__(self, params):
        """constructor
        """
        self.params = params  # all parameters
        self.device = params["device"]

        self.r0 = params["r0"]  # initial interest rate
        self.w0 = params["w0"]  # initial wealth
        self.T = params["T"]  # time horizon of the problem
        self.dt = params["dt"]  # time elapsed between periods
        self.sqrt_dt = np.sqrt(self.dt)
        self.t = T.linspace(0, self.T, int(self.T/self.dt+1))  # time space
        
        self.S_t = T.load(
            'sim_' + str(len(self.t)-1) + 'pers.pt', map_location=self.device)
        self.names_tickers = T.load(
            'names_tickers.pt', map_location=self.device)
        self.S0 = self.S_t[0,0,:].to(self.device)
        
    def __repr__(self):
        return f"<Environment T:{self.T}, dt:{self.dt}>"

    def __str__(self):
        descrp = "*** Portfolio allocation environment -- Saved dynamics ***\n"
        for key, val in self.params.items():
            descrp += '* {}: {}\n'.format(key, val)
        return descrp
        
    def reset(self, Nsims=1):
        """initialization of the environment with initial states

        Parameters
        ----------
        Nsims : int
            Number of realizations for the initial state

        Returns
        -------
        t_0, w_0 : torch.Tensor
            Tensors for the initial time, prices, and wealth
        """
        t_0 = T.zeros(Nsims, device=self.device)

        w_0 = self.w0 * T.ones(Nsims, device=self.device)

        return t_0, w_0

    def step(self, t_t, S_t, S_tp1, w_t, pi_t):
        """simulation engine

        Parameters
        ----------
        t_t, S_t, w_t, pi_t : torch.Tensor
            Tensors for the time, prices, wealth and policy

        Returns
        -------
        t_tp1, S_tp1, w_tp1 : torch.Tensor
            Tensors for the next time, prices, and wealth

        cost : torch.Tensor
            Tensor for the cost during this period
        """
        t_tp1 = t_t + self.dt  # update time step
       
        w_tp1 = w_t * T.sum(pi_t * S_tp1 / S_t, axis=-1)  # update wealth
        
        reward = w_tp1 - w_t  # compute reward

        return t_tp1, w_tp1, -T.clamp(reward, min=-5.0, max=5.0)
