#  Robust Reinforcement Learning with Dynamic Distortion Risk Measures

This Github repository contains the Python code and modules for the actor-critic algorithm described in our paper [Robust Reinforcement Learning with Dynamic Distortion Risk Measures](https://doi.org/10.48550/arXiv.2409.10096) by [Anthony Coache](https://anthonycoache.ca/) and [Sebastian Jaimungal](http://sebastian.statistics.utoronto.ca/).

Abstract: *In a reinforcement learning (RL) setting, the agent's optimal strategy heavily depends on her risk preferences and the underlying model dynamics of the training environment. These two aspects influence the agent's ability to make well-informed and time-consistent decisions when facing testing environments. In this work, we devise a framework to solve robust risk-aware RL problems where we simultaneously account for environmental uncertainty and risk with a class of dynamic robust distortion risk measures. Robustness is introduced by considering all models within a Wasserstein ball around a reference model. We estimate such dynamic robust risk measures using neural networks by making use of strictly consistent scoring functions, derive policy gradient formulae using the quantile representation of distortion risk measures, and construct an actor-critic algorithm to solve this class of robust risk-aware RL problems. We demonstrate the performance of our algorithm on a portfolio allocation example.*

***

### Instructions

* Open "hyperparams.py" and enter the appropriate parameters
* Open "main_train_cvar.py", enter the appropriate parameters, and run the file. This generates a folder with trained models for an agent with robustification
* Run "main_train_cvar_noeps.pr", enter the appropriate parameters, and run the file. This generates a folder with trained models for an agent without robustification
* Run "main_test_cvar.py" and run the file. This generates a folder with figures

For further details, please refer to our [paper](https://doi.org/10.48550/arXiv.2409.10096) or reach out to us. Thank you for your interest in our research work!

#### Authors

[Anthony Coache](https://anthonycoache.ca/) & [Sebastian Jaimungal](http://sebastian.statistics.utoronto.ca/)

***
