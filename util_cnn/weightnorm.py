# pylint: disable=C,R,E1101
"""
Like Batch Normalization but

- only compute the second moment (not the mean and not the variance)
- can be disabled by setting momentum to zero
- when disabled it uses only the running_moment, not the actual moment
"""
import torch


class WeightNorm(torch.nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features

        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_moment', torch.ones(self.num_features))

    def __repr__(self):
        return "{} (num_features={}, eps={}, momentum={})".format(
            self.__class__.__name__,
            self.num_features,
            self.eps,
            self.momentum)

    def forward(self, x):  # pylint: disable=W0221
        '''
        :param x: [batch, feature, ...]
        '''
        if self.training and self.momentum > 0:
            y = x.data ** 2
            y = y.view(x.size(0), self.num_features, -1).mean(-1).mean(0)  # [feature]
            self.running_moment = (1 - self.momentum) * self.running_moment + self.momentum * y  # pylint: disable=W0201

        factor = 1 / (self.running_moment + self.eps) ** 0.5
        x = x * torch.autograd.Variable(factor).view(1, -1, *(1,) * (x.dim() - 2))
        return x

    @staticmethod
    def set_all_momentum(net, momentum):
        for module in net.modules():
            if isinstance(module, WeightNorm):
                module.momentum = momentum


def test_batchnorm():
    bn = WeightNorm(4)

    x = torch.autograd.Variable(torch.randn(16, 4, 10, 10, 10))

    bn.momentum = 0.1
    bn(x)
    bn(x)

    bn.momentum = 0
    bn(x)
