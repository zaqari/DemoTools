import pandas as pd
import torch
import numpy as np
from scipy.stats import chi2

def tap(x):
    print(x.shape)
    return x

class wald():

    def __init__(self, betas, cov, k):
        super(wald, self).__init__()
        self.beta = betas.reshape(-1)
        self.cov = cov
        self.se = cov/np.sqrt(k-betas.shape[0])

    def __test_old(self, contrast, hypothesis: float=0.0):
        """
        performs a single contrast test!


        contrast: an ND matrix of shape 1 X self.beta.shape[0]
        hypothesis: a float indicating the hypothetical value
            are comparing the parameter estimates to. Default is 0.
        """
        variance_estimate = contrast.reshape(1,-1) @ self.vcov @ contrast.reshape(-1,1)
        hyp = (contrast @ self.beta) - hypothesis
        W = hyp.T @ np.linalg.inv(variance_estimate) @ hyp
        return W, 1-chi2.cdf(W, 1)

    def test(self, contrast, hypothesis: np.ndarray|float=0.0):
        """
        performs a single contrast test!


        contrast: an ND matrix of shape 1 X self.beta.shape[0]
        hypothesis: a float indicating the hypothetical value
            are comparing the parameter estimates to. Default is 0.
        """

        # if isinstance(self.beta, torch.FloatTensor):
        #     beta = self.beta.unsqueeze(-1)
        # else:
        #     beta = np.expand_dims(self.beta, axis=-1)
        beta = self.beta

        variance_estimate = contrast @ self.se @ contrast.transpose(1,0)
        hyp = (contrast @ beta) - hypothesis
        W = hyp.T @ np.linalg.inv(variance_estimate) @ hyp
        return W, contrast.shape[0], 1-chi2.cdf(W, contrast.shape[0])