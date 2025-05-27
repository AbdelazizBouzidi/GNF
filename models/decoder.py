import torch
import torch.nn as nn
from models.gnf_fields import RBFDecoder as GNF_RBFDecoder

class RBFDecoder(nn.Module):
    def __init__(self, config, encoder_feat_dim):
        super().__init__()
        num_levels = config['num_levels']
        num_features = encoder_feat_dim // num_levels if 'num_features' not in config else config['num_features']
        self.decoder = GNF_RBFDecoder(
            n_rbfs=config['n_rbfs'],
            out_dim=config['d_out'],
            num_levels=num_levels,
            num_features=num_features,
            basis_function=config['basis_function'],
            per_level=config.get('per_level', True),
            device=config.get('device', 'cuda')
        )

    def forward(self, x, features):
        return self.decoder(x)

class MLPDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config['num_layers']
        self.hidden_dim = config['hidden_dim']
        self.view_pe = config.get('view_pe', 0)
        self.fea_pe = config.get('fea_pe', 0)
        
        # Initialize MLP layers
        layers = []
        in_dim = config['d_in']
        for i in range(self.num_layers):
            out_dim = self.hidden_dim if i < self.num_layers - 1 else config['d_out']
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self.num_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, features):
        # x: [B, 3], features: [B, F]
        # Concatenate input and features
        x = torch.cat([x, features], dim=-1)
        return self.mlp(x)

def get_decoder(decoder_type, config, encoder_feat_dim):
    if decoder_type == 'rbf':
        return RBFDecoder(config, encoder_feat_dim)
    elif decoder_type == 'mlp':
        return MLPDecoder(config)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}") 