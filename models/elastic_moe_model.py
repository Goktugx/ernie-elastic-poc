"""
Elastic MoE Model

Three elasticity axes inspired by ERNIE 5.0:
- Elastic Depth: random number of active layers per step (3-6)
- Elastic Width: random number of active experts per step (4-16)
- Elastic Sparsity: random top-k routing per step (1-3)

Architecture: Conv feature extractor + N x MoE blocks + Classification head
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_layer import MoELayer


class MoEBlock(nn.Module):
    """Single MoE block: LayerNorm + MoE + residual connection."""

    def __init__(self, dim, moe_hidden_dim, num_experts=8, default_top_k=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.moe = MoELayer(
            input_dim=dim,
            hidden_dim=moe_hidden_dim,
            output_dim=dim,
            num_experts=num_experts,
            default_top_k=default_top_k,
        )

    def forward(self, x, top_k=None, num_active_experts=None):
        """
        Args:
            x: (batch, dim)
            top_k: elastic sparsity
            num_active_experts: elastic width
        Returns:
            output: (batch, dim)
            aux_loss: scalar
        """
        residual = x
        h = self.norm(x)
        moe_out, aux_loss = self.moe(h, top_k=top_k, num_active_experts=num_active_experts)
        return residual + moe_out, aux_loss


class ElasticMoEModel(nn.Module):
    """
    Elastic MoE model for CIFAR-10.

    Architecture:
    1. Conv feature extractor (fixed, not elastic)
    2. N MoE blocks (elastic depth/width/sparsity)
    3. Classification head

    Elastic configuration:
    - depth_choices: [3, 4, 5, 6] -- how many MoE blocks are active
    - width_choices: auto-scaled based on num_experts
    - sparsity_choices: [1, 2, 3] -- top-k routing
    """

    def __init__(
        self,
        num_classes=10,
        num_blocks=6,
        embed_dim=128,
        moe_hidden_dim=256,
        num_experts=8,
        default_top_k=2,
        depth_choices=None,
        width_choices=None,
        sparsity_choices=None,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        # Elastic config choices -- auto-scale with num_experts
        self.depth_choices = depth_choices or [3, 4, 5, 6]
        if width_choices is not None:
            self.width_choices = width_choices
        elif num_experts <= 8:
            self.width_choices = [4, 6, 8]
        else:
            # e.g. 16 experts -> [4, 8, 12, 16]
            self.width_choices = [
                num_experts // 4, num_experts // 2,
                3 * num_experts // 4, num_experts,
            ]
        self.sparsity_choices = sparsity_choices or [1, 2, 3]

        # --- Conv Feature Extractor ---
        # CIFAR-10: 3x32x32 -> feature vector
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),  # 128 x 4 x 4 = 2048
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.GELU(),
        )

        # --- MoE Blocks ---
        self.blocks = nn.ModuleList(
            [
                MoEBlock(embed_dim, moe_hidden_dim, num_experts, default_top_k)
                for _ in range(num_blocks)
            ]
        )

        # --- Classification Head ---
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        # Store current elastic config for logging
        self.current_config = None

    def sample_elastic_config(self):
        """Sample a random elastic configuration."""
        depth = random.choice(self.depth_choices)
        width = random.choice(self.width_choices)
        sparsity = random.choice(self.sparsity_choices)
        return {"depth": depth, "width": width, "sparsity": sparsity}

    def sample_sandwich_configs(self):
        """
        Sandwich rule: largest + smallest + random config.
        All three are trained each step.
        """
        largest = {
            "depth": max(self.depth_choices),
            "width": max(self.width_choices),
            "sparsity": max(self.sparsity_choices),
        }
        smallest = {
            "depth": min(self.depth_choices),
            "width": min(self.width_choices),
            "sparsity": min(self.sparsity_choices),
        }
        random_config = self.sample_elastic_config()
        return [largest, smallest, random_config]

    def forward(self, x, elastic_config=None):
        """
        Args:
            x: (batch, 3, 32, 32) -- CIFAR-10 images
            elastic_config: dict {"depth": int, "width": int, "sparsity": int}
                           If None, uses the largest configuration (inference default)
        Returns:
            logits: (batch, num_classes)
            aux_loss: total load balancing loss
            router_info: routing info per block (for visualization)
        """
        if elastic_config is None:
            elastic_config = {
                "depth": self.num_blocks,
                "width": self.num_experts,
                "sparsity": 2,
            }

        self.current_config = elastic_config
        depth = elastic_config["depth"]
        width = elastic_config["width"]
        sparsity = elastic_config["sparsity"]

        # Feature extraction
        h = self.feature_extractor(x)  # (B, embed_dim)

        # MoE blocks -- first 'depth' blocks are active, rest are skipped
        total_aux_loss = 0.0
        router_info = []

        for i in range(self.num_blocks):
            if i < depth:
                h, aux_loss = self.blocks[i](h, top_k=sparsity, num_active_experts=width)
                total_aux_loss += aux_loss
                router_info.append({
                    "block": i,
                    "probs": self.blocks[i].moe.last_router_probs,
                    "indices": self.blocks[i].moe.last_expert_indices,
                })
            # else: skip -- h unchanged (identity skip connection)

        # Classification
        logits = self.head(h)

        return logits, total_aux_loss, router_info

    def get_submodel_config(self, size="large"):
        """Dynamic sub-model configs based on actual num_blocks and num_experts."""
        n = self.num_experts
        b = self.num_blocks
        configs = {
            "large": {"depth": b, "width": n, "sparsity": 2},
            "medium": {"depth": max(1, b * 2 // 3), "width": max(1, 3 * n // 4), "sparsity": 2},
            "small": {"depth": max(1, b // 2), "width": max(1, n // 2), "sparsity": 1},
        }
        return configs[size]

    def count_active_params(self, elastic_config=None):
        """Count active parameters for a given elastic configuration."""
        if elastic_config is None:
            elastic_config = {
                "depth": self.num_blocks,
                "width": self.num_experts,
                "sparsity": 2,
            }

        # Feature extractor (always active)
        params = sum(p.numel() for p in self.feature_extractor.parameters())

        # Active blocks
        depth = elastic_config["depth"]
        width = elastic_config["width"]
        sparsity = elastic_config["sparsity"]

        for i in range(depth):
            block = self.blocks[i]
            # LayerNorm parameters
            params += sum(p.numel() for p in block.norm.parameters())
            # Router parameters
            params += sum(p.numel() for p in block.moe.router.parameters())
            # Active expert parameters (batched)
            params += block.moe.count_expert_params(width)

        # Classification head (always active)
        params += sum(p.numel() for p in self.head.parameters())

        return params
