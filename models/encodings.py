import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import N_to_reso
# import tinycudann as tcnn
import itertools
from typing import Sequence
from ops.interpolation import grid_sample_wrapper

def grid_mapping(positions, freq_bands, aabb, basis_mapping='sawtooth'):
    """Grid mapping function for DiF-Grid encoding"""
    aabbSize = max(aabb[1] - aabb[0])
    scale = aabbSize[..., None] / freq_bands
    
    if basis_mapping == 'triangle':
        pts_local = (positions - aabb[0]).unsqueeze(-1) % scale
        pts_local_int = ((positions - aabb[0]).unsqueeze(-1) // scale) % 2
        pts_local = pts_local / (scale / 2) - 1
        pts_local = torch.where(pts_local_int == 1, -pts_local, pts_local)
    elif basis_mapping == 'sawtooth':
        pts_local = (positions - aabb[0])[..., None] % scale
        pts_local = pts_local / (scale / 2) - 1
        pts_local = pts_local.clamp(-1., 1.)
    elif basis_mapping == 'sinc':
        pts_local = torch.sin((positions - aabb[0])[..., None] / (scale / np.pi) - np.pi / 2)
    elif basis_mapping == 'trigonometric':
        pts_local = (positions - aabb[0])[..., None] / scale * 2 * np.pi
        pts_local = torch.cat((torch.sin(pts_local), torch.cos(pts_local)), dim=-1)
    elif basis_mapping == 'x':
        pts_local = (positions - aabb[0]).unsqueeze(-1) / scale
    
    return pts_local

class DiFGridEncoder(nn.Module):
    """DiF-Grid encoding implementation using basis functions"""
    def __init__(self, 
                 input_dim: int,
                 basis_dims: list,
                 basis_resos: list,
                 freq_bands: list,
                 basis_mapping: str = 'sawtooth',
                 aabb: torch.Tensor = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.basis_dims = basis_dims
        self.basis_resos = basis_resos
        self.freq_bands = torch.tensor(freq_bands)
        self.basis_mapping = basis_mapping
        self.aabb = aabb if aabb is not None else torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        
        # Calculate output dimension
        self.output_dim = sum(basis_dims)
        
        # Initialize basis functions
        self.basis_functions = nn.ModuleList()
        for dim, reso in zip(basis_dims, basis_resos):
            if basis_mapping == 'grid':
                basis = nn.Parameter(torch.randn(1, dim, *([reso] * input_dim)))
            else:
                basis = nn.Parameter(torch.randn(1, dim))
            self.basis_functions.append(basis)

    def forward(self, x):
        # Apply grid mapping
        mapped_x = grid_mapping(x, self.freq_bands, self.aabb, self.basis_mapping)
        
        # Process through basis functions
        outputs = []
        for i, (basis, dim) in enumerate(zip(self.basis_functions, self.basis_dims)):
            if self.basis_mapping == 'grid':
                # Grid-based basis
                output = F.grid_sample(basis, mapped_x[..., i:i+1], 
                                     mode='bilinear', align_corners=True)
            else:
                # Other basis types
                output = basis * mapped_x[..., i:i+1]
            outputs.append(output.view(-1, dim))
            
        return torch.cat(outputs, dim=-1)

class HashGridEncoder(nn.Module):
    """HashGrid encoding using get_encoder function"""
    def __init__(self, 
                 input_dim: int,
                 num_levels: int,
                 num_features: int,
                 base_resolution: int,
                 desired_resolution: int,
                 log2_hashmap_size: int = 19,
                 per_level_scale: float = 1.38191):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.num_features = num_features
        self.base_resolution = base_resolution
        self.desired_resolution = desired_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.per_level_scale = per_level_scale
        
        # Initialize hash grid encoding using get_encoder
        self.encoding = get_encoder("hashgrid",
                                   input_dim=input_dim,
                                   num_levels=num_levels,
                                   level_dim=num_features,
                                   base_resolution=base_resolution,
                                   desired_resolution=desired_resolution)[0]
        
        self.output_dim = num_levels * num_features

    def forward(self, x):
       
        return self.encoding(x)

class MultiscaleTriplaneEncoder(nn.Module):
    def __init__(self, 
                 grid_nd: int = 2,
                 in_dim: int = 3,
                 feature_dim: int = 64,
                 base_res: int = 128,
                 multiscale_res_multipliers=[1, 2, 4]):
        super().__init__()

        self.grid_nd = grid_nd
        self.in_dim = in_dim
        self.coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        self.multiscale_res_multipliers = multiscale_res_multipliers

        num_levels = len(multiscale_res_multipliers)

        # Compute per-level feature dimensions
        base_feat_dim = feature_dim // num_levels
        remainder = feature_dim % num_levels
        feat_dims_per_level = [
            base_feat_dim + (1 if i < remainder else 0) 
            for i in range(num_levels)
        ]

        self.grids = nn.ModuleList()

        for level_idx, res_multiplier in enumerate(multiscale_res_multipliers):
            reso = [base_res * res_multiplier] * in_dim
            grid_coefs = nn.ParameterList()

            plane_feat_dim = feat_dims_per_level[level_idx]

            for coo_comb in self.coo_combs:
                grid_shape = [1, plane_feat_dim] + [reso[cc] for cc in coo_comb[::-1]]
                grid_param = nn.Parameter(torch.empty(grid_shape))
                nn.init.uniform_(grid_param, a=0.1, b=0.5)
                grid_coefs.append(grid_param)

            self.grids.append(grid_coefs)
        self.output_dim = feature_dim

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        multi_scale_feats = []
        eps = 1e-6

        # Convert to spherical coordinates
        norm = pts.norm(dim=-1, keepdim=True).clamp(min=eps)
        unit = pts / norm
        uv_r = ((norm - 0.75) / 1.5).squeeze(-1) 
        x, y, z = unit.unbind(dim=-1)
        theta = torch.asin(z.clamp(-1 + eps, 1 - eps)) / (torch.pi / 2)
        phi = torch.atan2(y, x) / torch.pi

        uv_coords = torch.stack([uv_r, phi, theta], dim=-1)

        for grid in self.grids:
            interp_space = 1.
            for grid_param, coo_comb in zip(grid, self.coo_combs):
                interp_out_plane = grid_sample_wrapper(
                    grid_param, uv_coords[..., coo_comb]
                ).view(-1, grid_param.shape[1])

                interp_space = interp_space * interp_out_plane

            multi_scale_feats.append(interp_space)

        return torch.cat(multi_scale_feats, dim=-1)

class HashGridPlanesEncoder(nn.Module):
    def __init__(self, 
                 grid_nd: int,
                 in_dim: int, 
                 num_levels: int,
                 num_features: int,
                 low_res: int,
                 high_res: int):
        super().__init__()

        assert grid_nd <= in_dim
        self.grid_nd = grid_nd
        self.in_dim = in_dim
        self.output_dim = num_levels * num_features
        self.coo_combs = list(itertools.combinations(range(in_dim), grid_nd))

        # Create hashgrid encoders for each plane
        self.plane_encoders = nn.ModuleList([
            get_encoder("hashgrid",
                       input_dim=grid_nd,
                       num_levels=num_levels,
                       level_dim=num_features,
                       base_resolution=low_res,
                       desired_resolution=high_res)[0]
            for _ in self.coo_combs
        ])

        self.output_dim = num_levels * num_features

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # Get features for each plane
        plane_features = [
            encoder(pts[..., coo_comb])
            for encoder, coo_comb in zip(self.plane_encoders, self.coo_combs)
        ]

        # Combine features from different planes
        combined_features = plane_features[0]
        for feats in plane_features[1:]:
            combined_features = combined_features + feats

        return combined_features

def get_encoder(encoding_type: str, **kwargs):
    """Factory function to create encoders"""
    if encoding_type == 'dif_grid':
        return DiFGridEncoder(**kwargs)
    elif encoding_type == 'hashgrid':
        return HashGridEncoder(**kwargs)
    elif encoding_type == 'kplanes':
        return MultiscaleTriplaneEncoder(**kwargs)
    elif encoding_type == 'hybrid-kplanes':
        return HashGridPlanesEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}") 