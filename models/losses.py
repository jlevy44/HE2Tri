import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia

# custom loss functions

class PerceptualLoss(nn.Module):
    def __init__(self,
            ff_net=torchvision.models.vgg16(pretrained=True),
            hooked_layers=[19, 23, 26, 30],
            layer_weights=None):
        super().__init__()
        self.ff_net = ff_net
        if torch.cuda.is_available():
            self.ff_net = self.ff_net.cuda()
        self.ff_net.eval()
        self.hooked_layers = hooked_layers
        if layer_weights == 'trainable':
            self.layer_weights = nn.Parameter(
                torch.ones(len(self.hooked_layers)))
        else:
            self.layer_weights = layer_weights
            if self.layer_weights is not None:
                assert len(layer_weights) == len(hooked_layers)
        def output_hook(layer, input, output):
            layer.output = output
        for i in hooked_layers:
            self.ff_net.features[i].register_forward_hook(output_hook)
    def forward(self, reconstructed, orig):
        with torch.no_grad():
            orig_out = self.ff_net(orig)
        features_orig = [
            self.ff_net.features[i].output for i in self.hooked_layers
        ]
        with torch.no_grad():
            recon_out = self.ff_net(reconstructed)
        features_recon = [
            self.ff_net.features[i].output for i in self.hooked_layers
        ]
        loss = 0.0
        for i in range(len(self.hooked_layers)):
            l = nn.L1Loss()(features_recon[i], features_orig[i])
            if self.layer_weights is not None:
                l *= self.layer_weights[i]
            loss += l
        return loss
    def __str__(self):
        return f'Perceptual loss\nLayer weights: {self.layer_weights}\
        \nHooked layers: {self.hooked_layers}'
