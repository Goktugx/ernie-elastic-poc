"""
Sub-Model Extraction

Extract different-sized sub-models from a single elastic training checkpoint.
This is the core advantage of elastic training: train once, deploy N models.
"""

import torch
import torch.nn as nn


class SubModelExtractor:
    """
    Extract sub-models from an elastic MoE checkpoint.

    Extraction logic:
    - depth: use first N blocks
    - width: use first M experts (router is sliced)
    - sparsity: controlled via top-k parameter at inference time
    """

    def __init__(self, elastic_model):
        self.model = elastic_model

    def extract(self, config):
        """
        Report sub-model info for a given configuration.

        The model itself is not modified -- elastic_config parameter controls
        which parts are active during forward pass. Here we report active
        parameter count and estimated inference cost.

        Args:
            config: {"depth": int, "width": int, "sparsity": int}

        Returns:
            info: sub-model statistics
        """
        depth = config["depth"]
        width = config["width"]
        sparsity = config["sparsity"]

        total_params = self.model.count_active_params(config)

        # Simplified FLOPs estimate
        embed_dim = self.model.embed_dim
        moe_hidden = self.model.blocks[0].moe.hidden_dim

        # Active experts per block = sparsity (top-k)
        flops_per_expert = embed_dim * moe_hidden * 2
        flops_moe_per_block = sparsity * flops_per_expert
        flops_moe_total = depth * flops_moe_per_block

        # Conv feature extractor FLOPs (fixed, approximate)
        flops_conv = 3 * 64 * 3 * 3 * 32 * 32 + 64 * 128 * 3 * 3 * 32 * 32
        flops_total = flops_conv + flops_moe_total

        info = {
            "config": config,
            "active_params": total_params,
            "estimated_flops": flops_total,
            "depth": depth,
            "width": width,
            "sparsity": sparsity,
        }

        return info

    def extract_all_presets(self):
        """Extract all preset sub-models and print a report."""
        results = {}
        for size in ["large", "medium", "small"]:
            config = self.model.get_submodel_config(size)
            info = self.extract(config)
            results[size] = info

        print("\n" + "=" * 70)
        print("SUB-MODEL EXTRACTION REPORT")
        print("=" * 70)
        for size, info in results.items():
            print(f"\n  {size.upper()}:")
            print(f"    Config: depth={info['depth']}, width={info['width']}, top-k={info['sparsity']}")
            print(f"    Active Params: {info['active_params']:,}")
            print(f"    Est. FLOPs: {info['estimated_flops']:,}")
        print("=" * 70)

        return results

    def compare_to_full(self):
        """Compare each sub-model to the full model."""
        full_config = {"depth": self.model.num_blocks, "width": self.model.num_experts, "sparsity": 2}
        full_params = self.model.count_active_params(full_config)

        results = {}
        for size in ["large", "medium", "small"]:
            config = self.model.get_submodel_config(size)
            sub_params = self.model.count_active_params(config)
            results[size] = {
                "config": config,
                "params": sub_params,
                "ratio": sub_params / full_params,
            }

        return results
