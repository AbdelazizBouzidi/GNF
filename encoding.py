import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

import torch
import torch.nn as nn
import itertools
from typing import Sequence
from ops.interpolation import grid_sample_wrapper

import torch
import torch.nn as nn
import itertools

class MultiscaleTriplaneEncoder(nn.Module):
    def __init__(self, 
                 grid_nd: int = 2,
                 in_dim: int = 3,
                 feature_dim: int = 64,
                 base_res: int = 128,
                 multiscale_res_multipliers=[1, 2, 4],
                 ):
        super().__init__()

        self.grid_nd = grid_nd
        self.in_dim = in_dim
        self.coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        self.multiscale_res_multipliers = multiscale_res_multipliers

        num_levels = len(multiscale_res_multipliers)

        # Compute per-level feature dimensions that sum up to the desired total feature dimension
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

            # Each plane in the same level has the SAME feature dimension as the final per-level dimension
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

        # Convert to spherical coordinates (θ, φ)
        norm = pts.norm(dim=-1, keepdim=True).clamp(min=eps)
        unit = pts / norm
        uv_r = ((norm - 0.75) / 1.5).squeeze(-1) 
        x, y, z = unit.unbind(dim=-1)
        theta = torch.asin(z.clamp(-1 + eps, 1 - eps)) / (torch.pi / 2)     # [0, pi]
        phi = torch.atan2(y, x) / torch.pi                         # [-1, 1]

        # breakpoint()
       
        uv_coords = torch.stack([uv_r, phi, theta], dim=-1)  # (B, 2)

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
                 high_res: int,
                 ):
        super().__init__()

        assert grid_nd <= in_dim
        self.grid_nd = grid_nd
        self.in_dim = in_dim
        self.output_dim = num_levels * num_features
        self.coo_combs = list(itertools.combinations(range(in_dim), grid_nd))

        # Create a separate hashgrid encoder for each coordinate combination (plane)
        self.plane_encoders = nn.ModuleList([
            get_encoder("hashgrid",
                        input_dim=grid_nd,
                        num_levels=num_levels,
                        level_dim=num_features,
                        base_resolution=low_res,
                        desired_resolution=high_res)[0]
            for _ in self.coo_combs
        ])

        # The output dim now is simply num_levels * num_features 
        # because planes features at same level are multiplied
        self.output_dim = num_levels * num_features

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # Compute features for each plane separately
        plane_features = [
            encoder(pts[..., coo_comb])  # [N, num_levels * num_features]
            for encoder, coo_comb in zip(self.plane_encoders, self.coo_combs)
        ]

        # Multiply features from different planes (element-wise product)
        # Start with features from the first plane
        combined_features = plane_features[0]
        for feats in plane_features[1:]:
            combined_features = combined_features + feats  # Hadamard product

        return combined_features
           
def get_encoder(encoding, input_dim=3, 
                multires=8, 
                degree=4,
                num_levels=16, level_dim=2, base_resolution=8, log2_hashmap_size=19, desired_resolution=256, align_corners=False,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim
    
    elif encoding == 'frequency':
        #encoder = FreqEncoder(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)
        from freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sphere_harmonics':
        from shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
    elif encoding == 'k-planes':
        encoder = MultiscaleTriplaneEncoder(input_dim - 1, input_dim, num_levels*level_dim)
        
    elif encoding == 'hybrid-k-planes':
        encoder = HashGridPlanesEncoder(input_dim - 1, input_dim, num_levels, level_dim, base_resolution, desired_resolution)    
        # encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)
    
    elif encoding == 'ash':
        from ashencoder import AshEncoder
        encoder = AshEncoder(input_dim=input_dim, output_dim=16, log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution)

    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim