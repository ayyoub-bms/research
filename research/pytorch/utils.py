import torch
import numpy as np

def get_device():
    kwargs = {}
    device_ = torch.device('cpu')
    if torch.cuda.is_available():
        print('Using CUDA')
        device_ = torch.device('cuda')
        kwargs = {'num_workers': 1, 'pin_memory': True}

    return device_, kwargs


class RegularizedModule(torch.nn.Module):
    """Class that implements regularizations of a neural network
    """
    def __init__(self):
        super(RegularizedModule, self).__init__()

    def _get_params(self, bias=False):
        params = []
        if bias:
            for param im self.parameters():
                params.append(param.view(-1))
            return torch.cat(params)

        for name, param in self.named_parameters():
            if 'bias' not in name:
                params.append(param.view(-1))
        return torch.cat(params)

    def lasso(self, lasso_param):
        params = self._get_params()
        return lasso_param * torch.linalg.norm(params, 1)

    def ridge(self, ridge_param):
        params = self._get_params()
        return ridge_param * torch.linalg.norm(params, 2)

    def elastic_net(self, lasso_param, ridge_param):
        return self.ridge(ridge_param) + self.lasso(lasso_param)
