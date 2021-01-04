"""
The source code for this file was taken from an amazing repository:
https://github.com/FrancescoSaverioZuppichini/A-journey-into-Convolutional-Neural-Network-visualization-
"""

import torch


class Base:
    def __init__(self, module):
        self.module = module
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Make sure that both model and all parameters are on the same device
        self.module.to(self.device)
        self.handles = []

    def clean(self):
        [h.remove() for h in self.handles]

    def __call__(self, inputs, layer, *args, **kwargs):
        return inputs, {}


class Weights(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def __call__(self, inputs, layer, *args, **kwargs):
        layer.register_forward_hook(self.hook)
        self.module(inputs.to(self.device))

        b, c, h, w = self.outputs.shape
        outputs = self.outputs.view(c, b, h, w)
        # reshape to make an array of images 1-Channel

        return outputs, {}
