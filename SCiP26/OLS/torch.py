import torch
import torch.nn as nn

class gaussian_regression(nn.Module):

    def __init__(self):
        super(gaussian_regression, self).__init__()
        self.xx = torch.FloatTensor()
        self.xy = torch.FloatTensor()

    def fit(self, x, y):

        if self.xx.shape[0] == 0:
            self.xx = x.t() @ x
            self.xy = x.t() @ y

        else:
            self.xx += x.t() @ x
            self.xy += x.t() @ y

    def forward(self, x, y):
        self.fit(x, y)

    def SSE(self, x, y):
        b = torch.linalg.inv(self.xx) @ self.xy
        return ((y - (x @ b))**2).sum()

    def estimates(self):
        xx_ = torch.linalg.inv(self.xx)
        return xx_ @ self.xy, xx_

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