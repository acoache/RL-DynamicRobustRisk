"""
Models:
    -- ANNs policy (Policy) and adversary (Adversary) components
    -- ANNs for the Q-function (QCVaR) and mean of costs-to-go (mu)

@date: Sept 2024
@author: Anthony Coache 
"""
# numpy
import numpy as np
# pytorch
import torch as T
import torch.nn as nn
from torch.nn.functional import silu, softmax, linear
import torch.optim as optim


def normalize_states(x, env):
    """batch normalization of features for ANNs
    """
    x[..., 0] = x[..., 0]  # time
    x[..., 1] = x[..., 1]/T.tensor(env.w0, device=env.device) - 1.0  # wealth
    x[..., 2:] = T.log(x[..., 2:]/env.S0)  # prices
    return x


def initialize_weights(layer):
    """initialization of weights and biases for all ANN
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)


class PosLinear(nn.Module):
    """layer applying a linear transformation with positive weights
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: T.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            T.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(T.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: T.Tensor) -> T.Tensor:
        return linear(input, self.weight**2, self.bias)
    
    
class CDFANN(nn.Module):
    def __init__(self, s_size, a_size, env, net_params):
        super(CDFANN, self).__init__()
        """constructor
        """
        self.s_size = s_size  # number of inputs for states
        self.a_size = a_size  # number of inputs for actions
        self.hidden_size = net_params["hidden_F"]  # number of hidden nodes
        self.n_layers = net_params["layers_F"]  # number of layers
        self.lr = net_params["lr_F"]  # learning rate
        self.gamma = net_params["gamma_F"]  # decay factor of the scheduler
        self.device = net_params["device"]  # device
        self.env = env  # environment (for normalisation purposes)

        # build all layers
        self.h1 = nn.Linear(self.s_size, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h3 = nn.Linear(self.a_size+self.hidden_size, self.hidden_size)
        self.h4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h5 = PosLinear(1+self.hidden_size, self.hidden_size)
        self.h_n = nn.ModuleList(
            [PosLinear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-4)])
        self.h_out = PosLinear(self.hidden_size, 1)

        # optimizers and schedulers
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=self.gamma)
        
        self.apply(initialize_weights)  # initialize weights and biases
        self.to(self.device)
        
    def forward(self, z, state_t, action_t):
        """forward propagation
        """
        s_norm = normalize_states(state_t, self.env)  # normalize features
        
        x = silu(self.h1(s_norm))
        x = silu(self.h2(x))
        x = silu(self.h3(T.cat((x, action_t), dim=-1)))
        x = silu(self.h4(x))
        x = T.tanh(self.h5(T.cat((x, z), dim=-1)))
        for layer in self.h_n:
            x = T.tanh(layer(x))

        return T.sigmoid(self.h_out(x))  # return CDF
    
    
class PolicyANN(nn.Module):
    def __init__(self, s_size, env, net_params):
        super(PolicyANN, self).__init__()
        """constructor
        """
        self.s_size = s_size  # number of inputs for states
        self.output_size = len(env.S0)  # number of outputs
        self.hidden_size = net_params["hidden_pi"]  # number of hidden nodes
        self.n_layers = net_params["layers_pi"]  # number of layers
        self.lr = net_params["lr_pi"]  # learning rate
        self.gamma = net_params["gamma_pi"]  # decay factor of the scheduler
        self.device = net_params["device"]  # device
        self.env = env  # environment (for normalisation purposes)

        # build all layers
        self.h1 = nn.Linear(self.s_size, self.hidden_size)
        self.h_n = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-1)])
        self.h_out = nn.Linear(self.hidden_size, self.output_size)
        
        # optimizers and schedulers
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=self.gamma)
        
        self.apply(initialize_weights)  # initialize weights and biases
        self.to(self.device)
        
    def forward(self, state_t):
        """forward propagation
        """
        s_norm = normalize_states(state_t, self.env)  # normalize features

        x = silu(self.h1(s_norm))
        for layer in self.h_n:
            x = silu(layer(x))
        x = self.h_out(x)
        
        return softmax(x, dim=-1)


class QCVaRANN(nn.Module):
    def __init__(self, s_size, a_size, env, net_params):
        super(QCVaRANN, self).__init__()
        """constructor
        """
        self.s_size = s_size  # number of inputs
        self.a_size = a_size  # number of inputs
        self.hidden_size = net_params["hidden_Q"]  # number of hidden nodes
        self.n_layers = net_params["layers_Q"]  # number of layers
        self.lr = net_params["lr_Q"]  # learning rate
        self.gamma = net_params["gamma_Q"]  # decay factor of the scheduler
        self.device = net_params["device"]  # device
        self.env = env  # environment (for normalisation purposes)
        
        # build all layers        
        self.h1_VaR = nn.Linear(self.s_size, self.hidden_size)
        self.h2_VaR = nn.Linear(self.hidden_size, self.hidden_size)
        self.h3_VaR = nn.Linear(self.a_size+self.hidden_size, self.hidden_size)
        self.h_n_VaR = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-3)])
        self.h_out_VaR = nn.Linear(self.hidden_size, 1)

        self.h1_CVaR = nn.Linear(self.s_size, self.hidden_size)
        self.h2_CVaR = nn.Linear(self.hidden_size, self.hidden_size)
        self.h3_CVaR = nn.Linear(self.a_size+self.hidden_size, self.hidden_size)
        self.h_n_CVaR = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-3)])
        self.h_out_CVaR = nn.Linear(self.hidden_size, 1)

        # optimizers and schedulers
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(optim.Adam([*self.h1_VaR.parameters(),
                                           *self.h2_VaR.parameters(),
                                           *self.h3_VaR.parameters(),
                                           *self.h_n_VaR.parameters(),
                                           *self.h_out_VaR.parameters()],
                                          lr=self.lr))
        self.schedulers.append(optim.lr_scheduler.StepLR(self.optimizers[0],
                                                         step_size=1,
                                                         gamma=self.gamma))
        self.optimizers.append(optim.Adam([*self.h1_CVaR.parameters(),
                                           *self.h2_CVaR.parameters(),
                                           *self.h3_CVaR.parameters(),
                                           *self.h_n_CVaR.parameters(),
                                           *self.h_out_CVaR.parameters()],
                                          lr=self.lr))
        self.schedulers.append(optim.lr_scheduler.StepLR(self.optimizers[1],
                                                         step_size=1,
                                                         gamma=self.gamma))
        
        self.apply(initialize_weights)  # initialize weights and biases
        self.to(self.device)

    def forward(self, state_t, action_t):
        """forward propagation
        """
        s_norm = normalize_states(state_t, self.env)  # normalize features
        
        y1 = silu(self.h1_VaR(s_norm.clone()))
        y1 = silu(self.h2_VaR(y1))
        y1 = silu(self.h3_VaR(T.cat((y1, action_t), -1)))
        for layer in self.h_n_VaR:
            y1 = silu(layer(y1))
        VaR = 8.0 * T.tanh(self.h_out_VaR(y1))
    
        y2 = silu(self.h1_CVaR(s_norm.clone()))
        y2 = silu(self.h2_CVaR(y2))
        y2 = silu(self.h3_CVaR(T.cat((y2, action_t), -1)))
        for layer in self.h_n_CVaR:
            y2 = silu(layer(y2))
        ExcessCVaR = T.clamp(self.h_out_CVaR(y2), min=0.0, max=5.0)
        
        return VaR, ExcessCVaR + VaR  # return Q-function (VaR, CVaR)


class MuANN(nn.Module):
    def __init__(self, s_size, a_size, env, net_params):
        super(MuANN, self).__init__()
        """constructor
        """
        self.s_size = s_size  # number of inputs
        self.a_size = a_size  # number of inputs
        self.hidden_size = net_params["hidden_Q"]  # number of hidden nodes
        self.n_layers = net_params["layers_Q"]  # number of layers
        self.lr = net_params["lr_Q"]  # learning rate
        self.gamma = net_params["gamma_Q"]  # decay factor of the scheduler
        self.device = net_params["device"]  # device
        self.env = env  # environment (for normalisation purposes)
        
        # build all layers
        self.h1 = nn.Linear(self.s_size, self.hidden_size)
        self.h2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h3 = nn.Linear(self.a_size+self.hidden_size, self.hidden_size)
        self.h_n = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-3)])
        self.h_out = nn.Linear(self.hidden_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1,
                                                   gamma=self.gamma)
        
        self.apply(initialize_weights)  # initialize weights and biases
        self.to(self.device)

    def forward(self, state_t, action_t):
        """forward propagation
        """
        s_norm = normalize_states(state_t, self.env)  # normalize features
        
        y0 = silu(self.h1(s_norm.clone()))
        y0 = silu(self.h2(y0))
        y0 = silu(self.h3(T.cat((y0, action_t), -1)))
        for layer in self.h_n:
            y0 = silu(layer(y0))
        
        return 5.0 * T.tanh(self.h_out(y0))  # return mu_F