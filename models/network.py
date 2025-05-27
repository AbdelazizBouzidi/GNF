import torch
import torch.nn as nn

class NeuralFieldNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        # x: [B, 3]
        # Get features from encoder
        features = self.encoder(x)
        # Decode features
        return self.decoder(x, features)

    def get_optparam_groups(self, lr_small, lr_large):
        # Get parameters for different learning rates
        params = []
        params.append({
            'params': self.encoder.parameters(),
            'lr': lr_large
        })
        params.append({
            'params': self.decoder.parameters(),
            'lr': lr_small
        })
        return params

    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }, path)

    def load(self, ckpt):
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder']) 