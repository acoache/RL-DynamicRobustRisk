"""
Dynamic risk measures:
    -- Expectation (mean), CVaR (CVaR)
    -- Computation of distortion function and (consistent) scoring function 

@date: Sept 2024
@author: Anthony Coache     
"""
# pytorch
import torch as T


class RiskMeasure():
    def __init__(self, params):
        """constructor
        """
        self.rm_type = params["type"]  # type of risk measure
        self.epsilon = params["epsilon"]  # tolerance
        self.beta = params["beta"]  # parameters for alpha-beta risk
        self.M = 15.0  # bound for random costs (for CVaR scoring function)
                
        if self.rm_type == 'CVaR':
            self.rm_label = self.rm_type + str(round(1.0-self.beta, 2)) \
                + "-eps" + str(round(self.epsilon, 5))
        else:
            raise ValueError("Unknown rm_type ('CVaR').")

    def __repr__(self):
        return f"<RiskMeasure rm_label:{self.rm_label}>"
    
    def __str__(self):
        return "Dynamic risk measure (" + self.rm_type + "): " + self.rm_label

    def get_gamma(self, u_grid):
        """get the distortion function of the risk measure
        """
        gamma = (u_grid >= self.beta) / (1-self.beta)

        return gamma
        
    def compute_scoring_Q(self, Q, obs_t, a_t, rvs):
        """compute a strictly consistent scoring function
        """
        VaRs_unsqz, CVaRs_unsqz = Q(obs_t.clone(), a_t.clone())
        VaRs = VaRs_unsqz.squeeze()
        CVaRs = CVaRs_unsqz.squeeze()
        
        scores = (T.log((CVaRs+self.M)/(rvs+self.M))
                  - CVaRs/(CVaRs+self.M)
                  + (1/((CVaRs+self.M)*(1-self.beta)))
                  * (VaRs*(1*(rvs <= VaRs)-self.beta) + rvs*(rvs > VaRs)))

        return T.mean(scores)

    def compute_scoring_mu(self, mu, obs_t, a_t, rvs):
        """compute a strictly consistent scoring function
        """
        means_unsqz = mu(obs_t.clone(), a_t.clone())
        means = means_unsqz.squeeze()
        
        return T.mean( (means - rvs)**2 )