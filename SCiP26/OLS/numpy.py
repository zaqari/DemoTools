import numpy as np
import torch
import torch.nn as nn

class gaussian_regression(nn.Module):

    def __init__(self):
        super(gaussian_regression, self).__init__()
        self.xx = np.array([])
        self.xy = np.array([])

    def fit(self, x, y):

        if self.xx.shape[0] == 0:
            self.xx = x.transpose() @ x
            self.xy = x.transpose() @ y

        else:
            self.xx += x.transpose() @ x
            self.xy += x.transpose() @ y

    def forward(self, x, y):
        self.fit(x, y)

    def estimates(self):
        xx_ = np.linalg.inv(self.xx)
        return xx_ @ self.xy, xx_

    def SSE(self, x, y):
        b = np.linalg.inv(self.xx) @ self.xy
        return ((y - (x @ b))**2).sum()

    def estimate_COV(self, x, y, return_SSE=False):
        b, cov = self.estimates()
        SSE = ((y - (x @ b))**2).sum()
        MSE = SSE / (len(y) - b.shape[0])

        if return_SSE:
            return MSE * cov, SSE
        else:
            return MSE * cov

    def use_external_MSE_to_calculate_COV(self, MSE):
        _, cov = self.estimates()
        return MSE * cov

    def save_checkpoint(self, file_name='regression_params.pt'):
        torch.save(
            {'xTx': self.xx, 'xTy': self.xy},
            file_name
        )

    def load_checkpoint(self, file_name='regression_params.pt'):
        ckpt = torch.load(file_name)
        self.xx = ckpt['xTx']
        self.xy = ckpt['xTy']