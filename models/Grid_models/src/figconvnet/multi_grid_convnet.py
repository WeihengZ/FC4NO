# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: S101,F722
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

try:
    from jaxtyping import Float
except ImportError:
    raise ImportError(
        "FIGConvUNet requires jaxtyping package, install using `pip install jaxtyping`"
    )

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from modulus.models.figconvnet.base_model import BaseModel
from modulus.models.figconvnet.components.encodings import SinusoidalEncoding
from modulus.models.figconvnet.components.mlp import MLP
from modulus.models.figconvnet.components.reductions import REDUCTION_TYPES
from modulus.models.figconvnet.geometries import (
    GridFeaturesMemoryFormat,
    PointFeatures,
)
from ..figconvnet.grid_feature_group import (
    GridFeatureConv2DBlocksAndIntraCommunication,
    GridFeatureGroup,
    GridFeatureGroupPadToMatch,
    GridFeatureGroupPool,
    GridFeatureGroupToPoint,
)
from modulus.models.figconvnet.point_feature_conv import (
    PointFeatureTransform,
)
from modulus.models.figconvnet.point_feature_grid_conv import (
    GridFeatureMemoryFormatConverter,
)
from ..figconvnet.point_feature_grid_ops import PointFeatureToGrid
from modulus.models.meta import ModelMetaData

memory_format_to_axis_index = {
    GridFeaturesMemoryFormat.b_xc_y_z: 0,
    GridFeaturesMemoryFormat.b_yc_x_z: 1,
    GridFeaturesMemoryFormat.b_zc_x_y: 2,
    GridFeaturesMemoryFormat.b_x_y_z_c: -1,
}

from ..fno.vt import ViT3D


class VerticesToPointFeatures(nn.Module):
    """
    VerticesToPointFeatures module converts the 3D vertices (XYZ coordinates) to point features.

    The module applies sinusoidal encoding to the vertices and optionally applies
    an MLP to the encoded vertices.
    """

    def __init__(
        self,
        embed_dim: int,
        out_features: Optional[int] = 32,
        use_mlp: Optional[bool] = True,
        pos_embed_range: Optional[float] = 2.0,
    ) -> None:
        super().__init__()
        self.pos_embed = SinusoidalEncoding(embed_dim, pos_embed_range)
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP(3 * embed_dim, out_features, [])

    def forward(self, vertices: Float[Tensor, "B N 3"]) -> PointFeatures:
        assert (
            vertices.ndim == 3
        ), f"Expected 3D vertices of shape BxNx3, got {vertices.shape}"
        vert_embed = self.pos_embed(vertices)
        if self.use_mlp:
            vert_embed = self.mlp(vert_embed)
        return PointFeatures(vertices, vert_embed)


@dataclass
class MetaData(ModelMetaData):
    name: str = "FIGConvUNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class FIGConvUNet(BaseModel):
    """Factorized Implicit Global Convolutional U-Net.

    The FIGConvUNet is a U-Net architecture that uses factorized implicit global
    convolutional layers to create U-shaped architecture. The advantage of using
    FIGConvolution is that it can handle high resolution 3D data efficiently
    using a set of factorized grids.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: List[int] = [512, 512],
        voxel_size: Optional[float] = None,
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int], Tuple[int, int, int], Tuple[float, float, float], Tuple[float, float, float]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128), (1,1,1), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128), (1,1,1), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2), (1,1,1), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp", "attn"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        drag_loss_weight: Optional[float] = None,
        pooling_type: Literal["attention", "max", "mean"] = "max",
        pooling_layers: List[int] = None,
        scale = 1,
    ):
        super().__init__(meta=MetaData())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        compressed_spatial_dims = []
        self.grid_feature_group_size = len(resolution_memory_format_pairs)
        self.point_feature_to_grids = nn.ModuleList()
        self.min_voxel_edge_length = torch.tensor([np.inf, np.inf, np.inf])
        
        '''
        Point to grid
        '''
        for mem_fmt, res, patch_res, abmax, abmin in resolution_memory_format_pairs:
            ab_length = torch.tensor(abmax) - torch.tensor(abmin)
            compressed_axis = memory_format_to_axis_index[mem_fmt]
            compressed_spatial_dims.append(res[compressed_axis])
            to_grid = nn.Sequential(
                PointFeatureToGrid(
                    in_channels=in_channels,
                    out_channels=hidden_channels[0],
                    aabb_max=abmax,
                    aabb_min=abmin,
                    voxel_size=voxel_size,
                    resolution=res,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_encode=use_rel_pos_embed,
                    pos_encode_dim=pos_encode_dim,
                    reductions=reductions,
                    neighbor_search_type=neighbor_search_type,
                    knn_k=knn_k,
                ),
                # GridFeatureMemoryFormatConverter(
                #     memory_format=mem_fmt,
                # ),
            )
            self.point_feature_to_grids.append(to_grid)
            # Compute voxel size
            voxel_size = ab_length / torch.tensor(res)
            self.min_voxel_edge_length = torch.min(
                self.min_voxel_edge_length, voxel_size
            )
        self.compressed_spatial_dims = compressed_spatial_dims
        self.convert_to_orig = GridFeatureMemoryFormatConverter(
            memory_format=GridFeaturesMemoryFormat.b_x_y_z_c
        )

        self.mlp = MLP(
            mlp_channels[0] * len(self.compressed_spatial_dims) * len(pooling_layers),
            mlp_channels[-1],
            mlp_channels,
            use_residual=True,
            activation=nn.GELU,
        )
        self.mlp_projection = nn.Linear(mlp_channels[-1], 1)
        # nn.Sigmoid(),

        '''
        Grid to point
        '''
        self.to_point = nn.ModuleList()
        for mem_fmt, res, patch_res, abmax, abmin in resolution_memory_format_pairs:
            back_to_grid= GridFeatureGroupToPoint(
                grid_in_channels=hidden_channels[0],
                point_in_channels=in_channels,
                out_channels=hidden_channels[0] * 2,
                grid_feature_group_size=1, # self.grid_feature_group_size,
                aabb_max=abmax,
                aabb_min=abmin,
                use_rel_pos=use_rel_pos,
                use_rel_pos_embed=use_rel_pos_embed,
                pos_embed_dim=pos_encode_dim,
                sample_method=to_point_sample_method,
                neighbor_search_type=neighbor_search_type,
                knn_k=knn_k,
                reductions=reductions,
            )
            self.to_point.append(back_to_grid)
        
        
        self.projection = nn.Sequential(
                    nn.Linear(hidden_channels[0] * 2, hidden_channels[0] * 2),
                    nn.LayerNorm(hidden_channels[0] * 2),
                    nn.GELU(),
                    nn.Linear(hidden_channels[0] * 2, hidden_channels[-1]),
                )

        self.pad_to_match = GridFeatureGroupPadToMatch()

        vertex_to_point_features = VerticesToPointFeatures(
            embed_dim=pos_encode_dim,
            out_features=hidden_channels[0],
            use_mlp=True,
            pos_embed_range=aabb_max[0] - aabb_min[0],
        )

        self.vertex_to_point_features = vertex_to_point_features
        if drag_loss_weight is not None:
            self.drag_loss_weight = drag_loss_weight

        '''
        VTs
        '''
        self.dim = hidden_channels[0]
        self.VisionTransformers = nn.ModuleList()
        for mem_fmt, res, patch_res, abmax, abmin in resolution_memory_format_pairs:
            vit = ViT3D(
                input_shape=res, patch_size=patch_res, 
                in_channels=self.dim, embed_dim=32,
                depth=3, num_heads=4, mlp_dim=64)
            self.VisionTransformers.append(vit)
        self.mllp = nn.Linear(self.dim, 1)
        

    def _grid_forward(self, point_features: PointFeatures):

        # grid_feature_group is a list of GridFeatures (length=3)
        # GridFeatures.vertices/features are tensors
        '''
        exact shape: [
            (torch.Size([1, 5, 150, 100, 3]), torch.Size([1, 80, 150, 100]))
            torch.Size([1, 250, 3, 100, 3]), torch.Size([1, 48, 250, 100])
            torch.Size([1, 250, 150, 2, 3]), torch.Size([1, 32, 250, 150])
        ]
        '''
        grid_feature_group = GridFeatureGroup(
            [to_grid(point_features) for to_grid in self.point_feature_to_grids]
        )

        # update name
        down_grid_feature_groups = [grid_feature_group]
        
        # # compute Drag prediction
        # pooled_feats = []
        # for grid_pool, layer in zip(self.grid_pools, self.pooling_layers):
        #     pooled_feats.append(grid_pool(down_grid_feature_groups[layer]))
        # if len(pooled_feats) > 1:
        #     pooled_feats = torch.cat(pooled_feats, dim=-1)
        # else:
        #     pooled_feats = pooled_feats[0]
        # drag_pred = self.mlp_projection(self.mlp(pooled_feats))

        '''
        input is a list of list of GridFeatures (with vertices and features)
        '''  
        # G shape == (B, M, N, L, F)
        for i in range(len(down_grid_feature_groups[0])):
            G = down_grid_feature_groups[0][i].features
            G = self.VisionTransformers[i](G)
            down_grid_feature_groups[0][i].features = G

        # compute drag prediction
        drag_pred = self.mllp(G)
        drag_pred = torch.mean(torch.mean(torch.mean(drag_pred, 1),1),1)

        # map the grid features back one by one
        grid_features = self.convert_to_orig(down_grid_feature_groups[0])

        return grid_features, drag_pred

    def forward(
        self,
        vertices: Float[Tensor, "B N 3"],
        features: Optional[Float[Tensor, "B N C"]] = None,
    ) -> Tensor:
        # structure the point features
        # point_features.features is a tensor of (1,M,F)
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)
        
        # point features -> grid features
        # operation on the grid features
        grid_features, drag_pred = self._grid_forward(point_features)

        # grid features -> point features
        # input grid features can be any shape
        Groups = []
        for i in range(len(grid_features)):
            Groups.append(
                GridFeatureGroup([grid_features[i]])
            )
        out_point_features = []
        for i in range(len(grid_features)):
            out_point_features.append(
                self.to_point[i](Groups[i], point_features)
            )

        # construct combined features
        out_total_features = 0 
        for i in range(len(grid_features)):
            out_total_features += out_point_features[i].features

        # map the features to final prediction
        out_total_features = self.projection(out_total_features)

        # final prediction is based on features
        return out_total_features, drag_pred
