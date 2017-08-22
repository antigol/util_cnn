# pylint: disable=C,R,E1101
"""
Like Batch Normalization but

- can be disabled by setting momentum to zero
- when disabled it uses only the running_moment, not the actual moment
"""
import torch


class AccBatchNorm(torch.nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features

        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.ones(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))

    def forward(self, x):  # pylint: disable=W0221
        '''
        :param x: [batch, feature, ...]
        '''
        if self.training and self.momentum > 0:
            mean = x.data.view(x.size(0), self.num_features, -1).mean(-1).mean(0)  # [feature]
            x2 = x.data ** 2
            x2 = x2.view(x.size(0), self.num_features, -1).mean(-1).mean(0)  # [feature]
            var = x2 - mean ** 2
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean  # pylint: disable=W0201
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var  # pylint: disable=W0201

        x = x - torch.autograd.Variable(self.running_mean).view(1, -1, *(1,) * (x.dim() - 2))
        factor = 1 / (self.running_var + self.eps) ** 0.5
        x = x * torch.autograd.Variable(factor).view(1, -1, *(1,) * (x.dim() - 2))
        return x

    @staticmethod
    def set_all_momentum(net, momentum):
        for module in net.modules():
            if isinstance(module, AccBatchNorm):
                module.momentum = momentum


def test_batchnorm():
    bn = AccBatchNorm(4)

    x = torch.autograd.Variable(torch.randn(16, 4, 10, 10, 10))

    bn.momentum = 0.1
    bn(x)
    bn(x)

    bn.momentum = 0
    bn(x)
