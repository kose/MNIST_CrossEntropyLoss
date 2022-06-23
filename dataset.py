# -*- mode: python -*-

from torchvision.datasets.mnist import MNIST

class subset_MNIST(MNIST):
    def __init__(self, nums=[1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        data =    [d for d, t in zip(self.data, self.targets) for n in nums if t == n]
        targets = [t for d, t in zip(self.data, self.targets) for n in nums if t == n]

        self.data = data
        self.targets = targets

### Local Variables: ###
### truncate-lines:t ###
### End: ###
