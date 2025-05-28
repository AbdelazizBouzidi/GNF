import torch
import torch.nn as nn
import numpy as np
import itertools

class DiFGridEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config.get('in_dim', 3)  # 3 for SDF, 2 for images
        self.basis_dims = config['basis_dims']
        self.basis_resos = config['basis_resos']
        self.freq_bands = config['freq_bands']
        self.basis_mapping = config['basis_mapping']
        self.align_corners = config.get('align_corners', True)
        
        # Initialize basis functions
        self.basis_functions = nn.ParameterList()
        for i in range(len(self.basis_dims)):
            if self.in_dim == 3:
                shape = (self.basis_dims[i],) + (self.basis_resos[i],) * 3  # (C,D,H,W)
            elif self.in_dim == 2:
                shape = (self.basis_dims[i],) + (self.basis_resos[i],) * 2  # (C,H,W)
            else:
                raise ValueError("DiFGridEncoder only supports in_dim 2 or 3")

            self.basis_functions.append(nn.Parameter(torch.randn(*shape)))
        self.output_dim = sum(self.basis_dims)

    def forward(self, x):
        # breakpoint()
        # x: [B, in_dim]
        features = []
        # Precompute full mapping tensor once (B,D,N_basis)
        aabb = torch.tensor([[-1.0] * self.in_dim, [1.0] * self.in_dim], device=x.device, dtype=x.dtype)
        freq_bands = torch.tensor(self.basis_resos, dtype=x.dtype, device=x.device)
        pts_local = _grid_mapping(x, freq_bands, aabb, self.basis_mapping)  # (B,D,N)
        
        for i, basis in enumerate(self.basis_functions):
            x_mapped = pts_local[..., i]  # (B,D)
            B = x_mapped.shape[0]

            if self.in_dim == 3:
                grid = x_mapped.view(B, 1, 1, 1, 3)  # (B,1,1,1,3)
                basis_expanded = basis.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B,C,D,H,W)
                feat = torch.nn.functional.grid_sample(
                    basis_expanded,
                    grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners
                ).squeeze(-1).squeeze(-1).squeeze(-1)  # (B,C)
            else:  # 2D
                grid = x_mapped.view(-1, 1, 1, 2)
                feat = torch.nn.functional.grid_sample(
                    basis.unsqueeze(0),  # (1,C,H,W)
                    grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners
                ).squeeze(0).squeeze(-1).squeeze(-1)  # (B,C)
            features.append(feat)
        return torch.cat(features, dim=1)

class KPlanesEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid_nd = config['grid_nd']
        self.feature_dim = config['feature_dim']
        self.base_res = config['base_res']
        self.multiscale_res_multipliers = config['multiscale_res_multipliers']
        self.align_corners = config.get('align_corners', True)
        
        # Initialize feature planes
        self.feature_planes = nn.ModuleList()
        for i in range(self.grid_nd):
            for j, mult in enumerate(self.multiscale_res_multipliers):
                res = self.base_res * mult
                self.feature_planes.append(
                    nn.Parameter(torch.randn(1, self.feature_dim, res, res))
                )

    def forward(self, x):
        # x: [B, 3]
        features = []
        for i in range(self.grid_nd):
            for j, mult in enumerate(self.multiscale_res_multipliers):
                res = self.base_res * mult
                # Get plane coordinates
                coords = x[:, [i, (i+1)%3]]
                # Map to plane resolution
                coords = (coords + 1) * (res - 1) / 2
                # Get features from plane
                feat = torch.nn.functional.grid_sample(
                    self.feature_planes[i * len(self.multiscale_res_multipliers) + j],
                    coords.view(-1, 1, 1, 2),
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=self.align_corners
                ).squeeze(0).squeeze(-1).squeeze(-1)
                features.append(feat)
        return torch.cat(features, dim=1)

class HybridKPlanesEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid_nd = config['grid_nd']  # 2 for planes
        self.in_dim = config.get('in_dim', 3)  # 3D input space
        self.num_levels = config['num_levels']
        self.level_dim = config['level_dim']
        self.base_resolution = config['base_resolution']
        self.desired_resolution = config['desired_resolution']
        self.align_corners = config.get('align_corners', True)
        
        # Get all possible 2D plane combinations
        self.coo_combs = list(itertools.combinations(range(self.in_dim), self.grid_nd))
        
        # Create a hashgrid encoder for each plane
        self.plane_encoders = nn.ModuleList([
            get_encoder("hashgrid",
                       input_dim=self.grid_nd,
                       num_levels=self.num_levels,
                       level_dim=self.level_dim,
                       base_resolution=self.base_resolution,
                       desired_resolution=self.desired_resolution)[0]
            for _ in self.coo_combs
        ])
        
        self.output_dim = self.num_levels * self.level_dim

    def forward(self, x):
        # x: [B, 3]
        # Get features for each plane using hashgrid
        plane_features = [
            encoder(x[..., coo_comb])  # [B, num_levels * level_dim]
            for encoder, coo_comb in zip(self.plane_encoders, self.coo_combs)
        ]
        
        # Combine features from different planes
        combined_features = plane_features[0]
        for feats in plane_features[1:]:
            combined_features = combined_features + feats  # Add features from different planes
            
        return combined_features

def get_encoder(encoder_type, config):
    if encoder_type == 'dif_grid':
        return DiFGridEncoder(config)
    elif encoder_type == 'kplanes':
        return KPlanesEncoder(config)
    elif encoder_type == 'hybrid-kplanes':
        return HybridKPlanesEncoder(config)
    elif encoder_type == 'hashgrid':
        from gridencoder import GridEncoder
        # breakpoint()
        return GridEncoder(
            input_dim=config.get('in_dim', 3),
            num_levels=config['num_levels'],
            level_dim=config['level_dim'],
            base_resolution=config['base_resolution'],
            desired_resolution=config['desired_resolution'],
            log2_hashmap_size=config.get('log2_hashmap_size', 19),
            gridtype='hash',
            align_corners=config.get('align_corners', True)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

def _grid_mapping(positions: torch.Tensor, freq_bands: torch.Tensor, aabb: torch.Tensor,
                  basis_mapping: str = 'sawtooth') -> torch.Tensor:
    """Replicates the mapping logic used in `models.gnf_fields.grid_mapping`.

    Parameters
    ----------
    positions : (B,D) tensor in world space (assumed in [-1,1]).
    freq_bands : (N,) tensor – one frequency per basis level.
    aabb : (2,D) tensor with scene bounds (we assume [-1,1] by default).
    basis_mapping : str – 'triangle' | 'sawtooth' | 'sinc' | 'trigonometric' | 'x'.
    Returns
    -------
    pts_local : (B,D,N) tensor – mapped coordinates per frequency band.
    """
    aabb_size = (aabb[1] - aabb[0]).max()
    scale = aabb_size[..., None] / freq_bands  # (N,)

    if basis_mapping == 'triangle':
        pts_local = (positions - aabb[0]).unsqueeze(-1) % scale
        pts_local_int = ((positions - aabb[0]).unsqueeze(-1) // scale) % 2
        pts_local = pts_local / (scale / 2) - 1
        pts_local = torch.where(pts_local_int == 1, -pts_local, pts_local)
    elif basis_mapping == 'sawtooth':
        pts_local = (positions - aabb[0]).unsqueeze(-1) % scale
        pts_local = pts_local / (scale / 2) - 1
        pts_local = pts_local.clamp(-1., 1.)
    elif basis_mapping == 'sinc':
        pts_local = torch.sin((positions - aabb[0]).unsqueeze(-1) / (scale / np.pi) - np.pi / 2)
    elif basis_mapping == 'trigonometric':
        pts_local = (positions - aabb[0]).unsqueeze(-1) / scale * 2 * np.pi
        pts_local = torch.cat((torch.sin(pts_local), torch.cos(pts_local)), dim=-1)
    elif basis_mapping == 'x':
        pts_local = (positions - aabb[0]).unsqueeze(-1) / scale
    else:
        raise ValueError(f'Unknown basis_mapping: {basis_mapping}')

    return pts_local 