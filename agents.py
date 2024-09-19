"""
Adversary-actor-critic algorithm:
    -- Robust risk-sensitive time-consistent optimization
    -- update_F(...), update_Q(...), update_mu(...), update_pi(...)
    
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
# misc
from time import time
import os
from utils import colors, rainbow
from datetime import datetime


class LearningAlgo():
    def __init__(self, env, RM,
                 Q, Q_target, pi, pi_target, F, F_target, mu, mu_target,
                 params):
        """constructor
        """
        self.env = env  # environment
        self.RM = RM  # risk measure

        self.pi = pi  # policy
        self.pi_target = pi_target  # target policy
        self.Q = Q  # Q-function
        self.Q_target = Q_target  # target Q-function
        self.F = F  # adversary
        self.F_target = F_target  # target adversary
        self.mu = mu  # mean of costs-to-go
        self.mu_target = mu_target  # target mean of costs-to-go

        self.device = self.pi.device  # PyTorch device
            
        self.tau = params["tau_target"]
        
        self.p_ex = params["p_ex"]
        self.min_p_ex = params["min_p_ex"]
        
        self.len_z_grid = params["len_z_grid"]
        self.z_s = T.linspace(-10.0, 10.0, self.len_z_grid)
        self.dz = self.z_s[1] - self.z_s[0]
        self.z_grid = self.z_s.reshape(self.len_z_grid, 1, 1).to(self.device)
        
        self.losses_F = []  # all losses for the adversary
        self.losses_Q = []  # all losses for the critic
        self.losses_pi = []  # all losses for the actor
        self.losses_mu = []  # all losses for the means
    
    def select_actions(self, t, S, w, pi, choose):
        """select an action according to the policy
        """
        # format to tensors
        obs = self.format_states_tensor(t.clone(), S.clone(), w.clone())

        # get actions from policy
        actions = pi(obs.clone())
        
        if choose == 'random':
            m = T.distributions.Dirichlet(0.05*T.ones(len(self.env.S0)))
            rnd_choice = T.multinomial(T.tensor([1-self.p_ex, self.p_ex], device=self.device),
                                       len(actions),
                                       replacement=True)
            actions[rnd_choice==1] = 0.75*actions[rnd_choice==1] + \
                0.25*m.sample((T.sum(rnd_choice),)).to(self.device)
        elif choose == 'best':
            actions = actions
        else:
            raise ValueError(
                "Type of action selection is unknown ('best' or 'random').")
        
        return actions.squeeze()
    
    def sim_trajs(self, Nsims, choose='random'):
        """simulate trajectories from the policy
        """
        # initialize tables for all trajectories
        t = T.zeros((Nsims, len(self.env.t)), dtype=T.float,
                    requires_grad=False, device=self.device)
        S = T.zeros((Nsims, len(self.env.t), len(self.env.S0)),
                    dtype=T.float, requires_grad=False, device=self.device)
        w = T.zeros((Nsims, len(self.env.t)), dtype=T.float,
                    requires_grad=False, device=self.device)
        actions = T.zeros((Nsims, len(self.env.t)-1, len(self.env.S0)),
                          dtype=T.float, requires_grad=False, device=self.device)
        costs = T.zeros((Nsims, len(self.env.t)-1), dtype=T.float,
                        requires_grad=False, device=self.device)
        
        t[:, 0], w[:, 0] = self.env.reset(Nsims)
        batch_idx = np.random.choice(len(self.env.S_t), size=Nsims, replace=False)
        S = self.env.S_t[batch_idx, :, :]
        
        for idx in range(len(self.env.t)-1):
            actions[:, idx, :] = self.select_actions(t[:, idx],
                                                     S[:, idx, :],
                                                     w[:, idx],
                                                     self.pi,
                                                     choose)
            
            t[:, idx+1], w[:, idx+1], costs[:, idx] = \
                self.env.step(t[:, idx],
                              S[:, idx, :],
                              S[:, idx+1, :],
                              w[:, idx],
                              actions[:, idx, :])
        trajs = {'t': t, 'S': S, 'w': w, 'costs': costs, 'actions': actions}
        
        return trajs
    
    def initial_estimates(self, Nsims, Nepochs, Nminibatch):
        """initial estimates for the ANNs
        """
        for epoch in range(300):
            with T.no_grad():
                # generate mini-batch of trajectories
                trajs = self.sim_trajs(Nsims=Nsims, choose='random')
            
            # estimate the Q-function
            self.update_Q_noeps(trajs, Nepochs, Nminibatch)
            for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data)
                
        for epoch in range(300):
            with T.no_grad():
                trajs = self.sim_trajs(Nsims=Nsims, choose='random')
                
            # estimate the CDF and first moment
            self.update_F(trajs, Nepochs, Nminibatch)
            self.update_mu(trajs, Nepochs, Nminibatch)
        
        for target_param, param in zip(self.F_target.parameters(), self.F.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.mu_target.parameters(), self.mu.parameters()):
            target_param.data.copy_(param.data)
        
        for epoch in range(300):
            with T.no_grad():
                trajs = self.sim_trajs(Nsims=Nsims, choose='random')
                
            # estimate the CDF
            self.update_Q(trajs, Nepochs, Nminibatch)
            for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def initial_estimates_noeps(self, Nsims, Nepochs, Nminibatch):
        """initial estimates for the ANNs
        """
        for epoch in range(300):
            with T.no_grad():
                # generate mini-batch of trajectories
                trajs = self.sim_trajs(Nsims=Nsims, choose='random')
            
            # estimate the Q-function
            self.update_Q_noeps(trajs, Nepochs, Nminibatch)
            for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def update_F(self, trajs, Nepochs, Nminibatch):
        """estimate the CDF of costs-to-go Z_{t}^{\theta} := c_{t} + Q_{t+1}
        """       
        # compute costs-to-go
        costs_to_go = self.compute_costs_to_go(trajs, self.Q, self.pi)
        
        for epoch in range(Nepochs):
            self.F.optimizer.zero_grad()  # zero out the gradient
            
            with T.no_grad():
                batch_idx = np.random.choice(len(trajs["t"]),
                                              size=Nminibatch,
                                              replace=False)
                
                t_idx = np.random.randint(len(self.env.t)-1)

                obs_t = self.format_states_tensor(trajs["t"][batch_idx, t_idx],
                                                  trajs["S"][batch_idx, t_idx, :],
                                                  trajs["w"][batch_idx, t_idx])
                
                Z_theta_t = costs_to_go[batch_idx, t_idx].repeat(self.len_z_grid,1)
                a_t = trajs["actions"][batch_idx, t_idx, :]
            
            # adversary prediction
            F_pred = self.F(self.z_grid.repeat(1, Nminibatch, 1),
                            obs_t.repeat(self.len_z_grid, 1, 1),
                            a_t.repeat(self.len_z_grid, 1, 1)).squeeze()
            
            # compute loss (z_grid, batch_idx, time)
            loss = T.mean(T.sum(
                (F_pred - 1.0*(Z_theta_t <= self.z_grid.repeat(1,Nminibatch,1)[...,0]))**2 * self.dz, axis=0))
            
            loss.backward()
            self.F.optimizer.step()
            self.losses_F.append(loss.detach().cpu().numpy())
        self.F.scheduler.step()
                
    def update_mu(self, trajs, Nepochs, Nminibatch):
        """estimate the first moment of costs-to-go
        """
        # compute costs-to-go
        costs_to_go = self.compute_costs_to_go(trajs, self.Q, self.pi)
        
        for epoch in range(Nepochs):

            self.mu.optimizer.zero_grad()  # zero out the gradient

            with T.no_grad():
                batch_idx = np.random.choice(len(trajs["t"]),
                                             size=Nminibatch,
                                             replace=False)
                
                t_idx = np.random.randint(len(self.env.t)-1)
                
                # observations s_{t} as a formatted tensor
                obs_t = self.format_states_tensor(trajs["t"][batch_idx, t_idx],
                                                  trajs["S"][batch_idx, t_idx, :],
                                                  trajs["w"][batch_idx, t_idx])
                Z_theta_t = costs_to_go[batch_idx, t_idx]
                a_t = trajs["actions"][batch_idx, t_idx, :]
            
            loss = self.RM.compute_scoring_mu(mu=self.mu,
                                              obs_t=obs_t,
                                              a_t=a_t,
                                              rvs=Z_theta_t)
            
            loss.backward()
            self.mu.optimizer.step()
            self.losses_mu.append(loss.detach().cpu().numpy())
        self.mu.scheduler.step()
    
    def update_Q(self, trajs, Nepochs, Nminibatch):
        """estimate the Q-function
        """
        # compute costs-to-go
        costs_to_go = self.compute_costs_to_go(trajs, self.Q_target, self.pi_target)
        
        for epoch in range(Nepochs):

            for optimizer in self.Q.optimizers:  # zero out the gradient
                optimizer.zero_grad()

            with T.no_grad():
                batch_idx = np.random.choice(len(trajs["t"]),
                                             size=Nminibatch,
                                             replace=False)
                
                t_idx = np.random.randint(len(self.env.t)-1)
                
                # observations s_{t} as a formatted tensor
                obs_t = self.format_states_tensor(trajs["t"][batch_idx, t_idx],
                                                  trajs["S"][batch_idx, t_idx, :],
                                                  trajs["w"][batch_idx, t_idx])
                Z_theta_t = costs_to_go[batch_idx, t_idx]
                a_t = trajs["actions"][batch_idx, t_idx, :]
                
                # generate conditional uniform rvs from adversary
                U_t = self.F_target(Z_theta_t.unsqueeze(-1),
                                    obs_t.clone(),
                                    a_t).squeeze()
                                
                # compute saddle-points for each state
                mu = self.mu_target(obs_t.clone(), a_t.clone()).detach().squeeze()
                lmbd, b_lmbd = self.get_saddlepoint(self.F_target, obs_t.clone(), a_t.clone(), mu)
                
                # generate distorted costs from optimal quantile function for each state
                Z_phi_t = mu + (lmbd*Z_theta_t+self.RM.get_gamma(U_t) - (lmbd*mu+1.0)) / b_lmbd
                
            loss = self.RM.compute_scoring_Q(Q=self.Q,
                                              obs_t=obs_t,
                                              a_t=a_t,
                                              rvs=Z_phi_t)
            
            loss.backward()
            for optimizer in self.Q.optimizers:
                optimizer.step()
            self.losses_Q.append(loss.detach().cpu().numpy())
        for scheduler in self.Q.schedulers:
            scheduler.step()
                        
    def update_Q_noeps(self, trajs, Nepochs, Nminibatch):
        """estimate the Q-function without uncertainty
        """
        # compute costs-to-go
        costs_to_go = self.compute_costs_to_go(trajs, self.Q_target, self.pi_target)
        
        for epoch in range(Nepochs):

            for optimizer in self.Q.optimizers:  # zero out the gradient
                optimizer.zero_grad()

            with T.no_grad():
                batch_idx = np.random.choice(len(trajs["t"]),
                                             size=Nminibatch,
                                             replace=False)
                
                t_idx = np.random.randint(len(self.env.t)-1)
                
                # observations s_{t} as a formatted tensor
                obs_t = self.format_states_tensor(trajs["t"][batch_idx, t_idx],
                                                  trajs["S"][batch_idx, t_idx, :],
                                                  trajs["w"][batch_idx, t_idx])
                Z_theta_t = costs_to_go[batch_idx, t_idx]
                
                a_t = trajs["actions"][batch_idx, t_idx, :]
                               
            loss = self.RM.compute_scoring_Q(Q=self.Q,
                                              obs_t=obs_t,
                                              a_t=a_t,
                                              rvs=Z_theta_t)
            
            loss.backward()
            for optimizer in self.Q.optimizers:
                optimizer.step()
            self.losses_Q.append(loss.detach().cpu().numpy())
        for scheduler in self.Q.schedulers:
            scheduler.step()

    def update_pi(self, Nepochs, Nminibatch):
        """update the policy
        """        
        for epoch in range(Nepochs):
            self.pi.optimizer.zero_grad()  # zero out the gradient
            
            with T.no_grad():
                # generate mini-batch of trajectories
                trajs = self.sim_trajs(Nsims=Nminibatch, choose='best')
                
                # compute costs-to-go
                Z_theta_t = self.compute_costs_to_go(trajs, self.Q, self.pi)
                
                # observations s_{t} as a formatted tensor
                obs_t = self.format_states_tensor(trajs["t"][:, :-1],
                                                  trajs["S"][:, :-1, :],
                                                  trajs["w"][:, :-1]).detach()
                a_t = trajs["actions"]
                
                # compute saddle-points for each state
                mu = self.mu(obs_t.clone(), a_t.clone()).detach().squeeze()
                lmbd, b_lmbd = self.get_saddlepoint(self.F, obs_t.clone(), a_t.clone(), mu)
                
            # gradient of F wrt its inputs
            Z_grad = Z_theta_t.unsqueeze(-1).clone().requires_grad_()
            
            grad_F_Z_inputs = self.F(Z_grad, obs_t.clone(), a_t)
            f_Z_theta = T.autograd.grad(inputs=Z_grad,
                                        outputs=grad_F_Z_inputs.sum(),
                                        grad_outputs=None)[0].detach().squeeze()
            
            # forward propagation
            a_theta = self.pi(obs_t.clone())
            
            # gradient of F wrt policy parameters
            grad_F_Z_theta = self.F(Z_theta_t.unsqueeze(-1).clone(),
                                    obs_t.clone(),
                                    a_theta).squeeze()

            penalty_theta = ((b_lmbd-lmbd)*(Z_theta_t-mu) + 1.0) * \
                grad_F_Z_theta / T.clamp(f_Z_theta, min=5e-4)
                
            # gradient of Q-function
            Q_theta = self.get_Q_function(self.Q, obs_t.clone(), a_theta)
            
            loss_theta = T.mean(Q_theta - (b_lmbd-lmbd)*penalty_theta/b_lmbd)
            loss_theta.backward()
            self.pi.optimizer.step()
            self.losses_pi.append(loss_theta.detach().cpu().numpy())
        self.pi.scheduler.step()
    
    def update_pi_noeps(self, Nepochs, Nminibatch):
        """update the policy without uncertainty
        """        
        for epoch in range(Nepochs):
            self.pi.optimizer.zero_grad()  # zero out the gradient
            
            with T.no_grad():
                # generate mini-batch of trajectories
                trajs = self.sim_trajs(Nsims=Nminibatch, choose='best')
                                
                # observations s_{t} as a formatted tensor
                obs_t = self.format_states_tensor(trajs["t"][:, :-1],
                                                  trajs["S"][:, :-1, :],
                                                  trajs["w"][:, :-1]).detach()
            
            # forward propagation
            a_theta = self.pi(obs_t.clone())
                
            # gradient of Q-function
            Q_theta = self.get_Q_function(self.Q, obs_t.clone(), a_theta)
            
            loss_theta = T.mean(Q_theta)
            loss_theta.backward()
            self.pi.optimizer.step()
            self.losses_pi.append(loss_theta.detach().cpu().numpy())
        self.pi.scheduler.step()
    
    def format_states_tensor(self, t, S, w):
        """format the features of the ANNs into a single tensor
        """
        return T.cat((t.unsqueeze(-1), w.unsqueeze(-1), S), -1)
    
    def compute_costs_to_go(self, trajs, Q, pi):
        """compute the costs-to-go and the z_grid for the adversary
        """
        # observations s_{t+1} as a formatted tensor
        obs_tp1 = self.format_states_tensor(trajs["t"][:, 1:-1],
                                            trajs["S"][:, 1:-1, :],
                                            trajs["w"][:, 1:-1])
        
        costs_to_go = trajs["costs"].detach().clone()  # get costs c_{t}
        a_tp1 = pi(obs_tp1.clone())  # get actions a_{t+1}
        Q_tp1 = self.get_Q_function(Q, obs_tp1, a_tp1)  # get Q_{t+1}
        costs_to_go[:, :-1] += Q_tp1  # compute c_{t} + Q_{t+1}
        
        return costs_to_go
    
    def get_Q_function(self, Q, obs_t, a_t):
        """get the value function Q_{t} from the ANN
        """
        VaR_pred_t, CVaR_pred_t = Q(obs_t.clone(), a_t.clone())
        Q_function = CVaR_pred_t.squeeze()
        
        return Q_function
    
    def get_saddlepoint(self, F, obs_t, a_t, mu_F):
        """compute saddle-points (optimal quantile function) for each state
        """        
        # create grids        
        if len(obs_t.shape) == 2:
            u_s = F(self.z_grid.repeat(1, len(obs_t), 1),
                    obs_t.repeat(self.len_z_grid, 1, 1),
                    a_t.repeat(self.len_z_grid, 1, 1)).squeeze()  # F(Z)'s
            F_s = self.z_grid.repeat(1, len(obs_t), 1).squeeze()  # \breve{F}(F(Z))'s
            g_s = self.RM.get_gamma(u_s)  # \gamma(F(Z))'s
        else:
            u_s = F(self.z_grid.unsqueeze(-1).repeat(1, len(obs_t), len(self.env.t)-1, 1),
                    obs_t.repeat(self.len_z_grid, 1, 1, 1),
                    a_t.repeat(self.len_z_grid, 1, 1, 1)).squeeze()  # F(Z)'s
            F_s = self.z_grid.repeat(1, len(obs_t), len(self.env.t)-1)  # \breve{F}(F(Z))'s
            g_s = self.RM.get_gamma(u_s)  # \gamma(F(Z))'s
        
        # normalize integral grid
        u_diff = T.clamp(T.diff(u_s, axis=0), min=0.0) + 1e-6*(T.diff(u_s, axis=0) < 0)
        u_diff /= T.sum(u_diff, axis=0)
        
        # compute var(\breve{F}), var(\gamma), cov(\breve{F}, \gamma)
        sigma_F = T.sum(u_diff*0.5*
                        ((F_s-mu_F)[:-1,...]**2+(F_s-mu_F)[1:,...]**2),
                        axis=0)
        sigma_g = T.sum(u_diff*0.5*
                        ((g_s-1.0)[:-1,...]**2+(g_s-1.0)[1:,...]**2),
                        axis=0)
        sigma_Fg = T.sum(u_diff*0.5*
                         (((F_s-mu_F)*(g_s-1.0))[:-1,...]+((F_s-mu_F)*(g_s-1.0))[1:,...]),
                         axis=0)
        
        # get analytical saddle-points
        K = sigma_F - 0.5*self.RM.epsilon**2
        
        delta = 4*(sigma_Fg**2 + sigma_F*(sigma_F*sigma_Fg**2 - K**2*sigma_g) /
                   (self.RM.epsilon**2 * (0.25*self.RM.epsilon**2 - sigma_F)))
        
        lmbd = (-2*sigma_Fg + T.sqrt(T.clamp(delta, min=0.0)))/(2*sigma_F)
        lmbd[K < (sigma_Fg * T.sqrt(sigma_F) / T.sqrt(sigma_g))] = 0.0
        
        b_lmbd_num = lmbd**2*sigma_F + 2*lmbd*(sigma_Fg) + sigma_g
        b_lmbd = T.sqrt(b_lmbd_num) / T.sqrt(sigma_F)
        
        return lmbd, b_lmbd
    
    def load_models(self, repo, optim=True):
        """load parameters of the (trained) ANNs in main/target networks
        """
        if not os.path.exists(repo):
            raise IOError("The specified repository does not exist.")
        
        checkpoint = T.load(repo + '/checkpoint.pt', map_location=self.device)
        self.pi.load_state_dict(checkpoint['pi_model'])
        self.Q.load_state_dict(checkpoint['Q_model'])
        self.F.load_state_dict(checkpoint['F_model'])
        self.mu.load_state_dict(checkpoint['mu_model'])
        
        self.pi_target.load_state_dict(checkpoint['pi_target_model'])
        self.Q_target.load_state_dict(checkpoint['Q_target_model'])
        self.F_target.load_state_dict(checkpoint['F_target_model'])
        self.mu_target.load_state_dict(checkpoint['mu_target_model'])
        
        if optim:
            self.pi.optimizer.load_state_dict(checkpoint['pi_optim'])
            for idx, optimizer in enumerate(self.Q.optimizers):
                optimizer.load_state_dict(checkpoint['Q_optim' + str(idx)])
            self.F.optimizer.load_state_dict(checkpoint['F_optim'])
            self.mu.optimizer.load_state_dict(checkpoint['mu_optim'])

    def save_models(self, repo, timestamp=False):
        """save parameters of the (trained) ANNs
        """
        if not os.path.exists(repo):
            raise IOError("The specified repository does not exist.")
        
        if timestamp:
            now = datetime.now()
            stamp = '-'+str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)
        else:
            stamp = ""
        
        dict_checkpoint = {
            'pi_model': self.pi.state_dict(),
            'pi_target_model': self.pi_target.state_dict(),
            'pi_optim': self.pi.optimizer.state_dict(),
            'Q_model': self.Q.state_dict(),
            'Q_target_model': self.Q_target.state_dict(),
            'F_model': self.F.state_dict(),
            'F_target_model': self.F_target.state_dict(),
            'F_optim': self.F.optimizer.state_dict(),
            'mu_model': self.mu.state_dict(),
            'mu_target_model': self.mu_target.state_dict(),
            'mu_optim': self.mu.optimizer.state_dict()
            }
        
        for idx, optimizer in enumerate(self.Q.optimizers):
            dict_checkpoint['Q_optim' + str(idx)] = optimizer.state_dict()
        
        T.save(dict_checkpoint, repo + '/checkpoint' + stamp + '.pt')

    def set_train_mode(self, train=True):
        """set training/eval mode for ANNs
        """
        if train:
            self.pi.train()
            self.Q.train()
            self.F.train()
            self.mu.train()
        else:
            self.pi.eval()
            self.Q.eval()
            self.F.eval()
            self.mu.eval()
         
    def decay_exploration(self):
        """decay the parameter for the exploratory policy
        """
        self.p_ex = np.maximum(self.p_ex*0.99999, self.min_p_ex)
    
    def update_targets(self):
        """ update of all target neural networks
        """
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.pi_target.parameters(), self.pi.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.F_target.parameters(), self.F.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.mu_target.parameters(), self.mu.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def print_diagnostic(self, epoch, start_time):
        """print all statistics during training phase
        """
        print('*** Epoch = ', str(epoch), ' completed, Duration = ',
              "{:.2f}".format((time() - start_time)/60), ' mins ***\n')
        print('pi loss: ', str(np.round(np.mean(self.losses_pi[-10:]), 6)),
              '; Q loss: ', str(np.round(np.mean(self.losses_Q[-1000:]), 6)),
              '; mu loss: ', str(np.round(np.mean(self.losses_mu[-1000:]), 6)),
              '; F loss: ', str(np.round(np.mean(self.losses_F[-1000:]), 6)))
        print('lr_pi: ', "{:.3e}".format(self.pi.optimizer.param_groups[0]["lr"]),
              '; lr_Q: ', "{:.3e}".format(self.Q.optimizers[0].param_groups[0]["lr"]),
              '; lr_F: ', "{:.3e}".format(self.F.optimizer.param_groups[0]["lr"]),
              '; exploration: ', "{:.4f}".format(self.p_ex), '\n')
    
    def clean_losses(self):
        del self.losses_pi[:(len(self.losses_pi) - 10)]
        del self.losses_Q[:(len(self.losses_Q) - 1000)]
        del self.losses_F[:(len(self.losses_F) - 1000)]
        del self.losses_mu[:(len(self.losses_mu) - 1000)]
        
    def plot_diagnostics(self, repo, Nsims=5_000, seed=1234):
        """plot illustrations to evaluate the current (optimal) policy
        """
        if not os.path.exists(repo):
            raise IOError("The specified repository does not exist.")

        with T.no_grad():
            T.manual_seed(seed)
            np.random.seed(seed)
            trajs = self.sim_trajs(Nsims=Nsims, choose='best')

            rewards_total = -1 * T.sum(trajs["costs"], axis=1).cpu().numpy()
            pis_paths = trajs["actions"].cpu().numpy()

        # set grids for PnL
        reward_grid = np.linspace(-self.env.w0, 2*self.env.w0, 100)
        
        nrows = 2  # number of rows
        ncols = 1  # number of columns

        # figure parameters
        plt.rcParams.update({'font.size': 16, 'figure.figsize': (14, 7)})
        plt.rc('axes', labelsize=20)
        fig, axes = plt.subplots(nrows, ncols)
        
        axes[0].stackplot(self.env.t[:-1].cpu().numpy(),
                          np.mean(pis_paths[:,::-1,:], axis=0).transpose(),
                          labels=self.env.names_tickers[::-1],
                          colors=rainbow)
        axes[0].set_xlabel(r"$t$")
        axes[0].set_ylabel(r"$\pi_{t}^{(i)}$")
        axes[0].set_xlim(0,self.env.t[-2])
        axes[0].set_ylim(0.0,1.0)
        handles, labels = axes[0].get_legend_handles_labels()

        axes[1].hist(x=rewards_total, bins=reward_grid,
                     alpha=0.4, color=colors[0], density=True)
        kde = gaussian_kde(rewards_total, bw_method='silverman')
        axes[1].plot(reward_grid, kde(reward_grid),
                     color=colors[0], linewidth=1.5)
        axes[1].axvline(x=np.quantile(rewards_total, 0.1),
                        linestyle='dashed', color=colors[0], linewidth=1.0)
        axes[1].axvline(x=np.quantile(rewards_total, 0.9),
                        linestyle='dashed', color=colors[0], linewidth=1.0)
        axes[1].set_xlabel("P&L")
        axes[1].set_ylabel("Density")
        axes[1].set_xlim(np.min(reward_grid), np.max(reward_grid))

        fig.tight_layout()
        fig.legend(handles[::-1], labels[::-1], loc=7)
        fig.subplots_adjust(right=0.85)
        now = datetime.now()
        plt.savefig(repo + '/plot-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.png',
                    transparent=False, bbox_inches='tight')
        plt.clf()
        plt.close()
        