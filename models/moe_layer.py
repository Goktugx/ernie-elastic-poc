"""
Mixture of Experts (MoE) Layer -- GPU-optimized batched implementation.

All experts computed in parallel via einsum (no sequential loops).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Linear gate that produces expert scores."""

    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x, num_active_experts=None):
        logits = self.gate(x)
        if num_active_experts is not None:
            logits = logits[:, :num_active_experts]
        return logits


class MoELayer(nn.Module):
    """
    GPU-optimized MoE layer with batched expert computation.

    Instead of looping over experts one by one, all expert weights are
    stored as batched tensors and computed in parallel via torch.einsum.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_experts=8,
        default_top_k=2,
        aux_loss_weight=0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.default_top_k = default_top_k
        self.aux_loss_weight = aux_loss_weight
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Batched expert weights: all experts in single tensors
        self.w1 = nn.Parameter(torch.empty(num_experts, input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(num_experts, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, output_dim))
        self.b2 = nn.Parameter(torch.empty(num_experts, output_dim))
        self._init_weights()

        self.router = Router(input_dim, num_experts)

        # Store last routing info for visualization
        self.last_router_probs = None
        self.last_expert_indices = None

    def _init_weights(self):
        """Match nn.Linear default initialization (kaiming_uniform)."""
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            bound1 = 1 / math.sqrt(self.input_dim)
            nn.init.uniform_(self.b1[i], -bound1, bound1)

            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            bound2 = 1 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(self.b2[i], -bound2, bound2)

    def forward(self, x, top_k=None, num_active_experts=None):
        """
        Args:
            x: (batch, input_dim)
            top_k: how many experts active per sample (elastic sparsity)
            num_active_experts: how many experts available (elastic width)
        Returns:
            output: (batch, output_dim)
            aux_loss: load balancing loss (scalar)
        """
        top_k = top_k or self.default_top_k
        num_active = num_active_experts or self.num_experts
        top_k = min(top_k, num_active)

        # Router scores
        router_logits = self.router(x, num_active_experts=num_active)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, num_active)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Store for visualization
        self.last_router_probs = router_probs.detach()
        self.last_expert_indices = top_k_indices.detach()

        # --- Parallel expert computation (no loops!) ---
        # x: (B, D) @ w1[:E]: (E, D, H) -> h: (B, E, H)
        h = torch.einsum('bd,edh->beh', x, self.w1[:num_active]) + self.b1[:num_active]
        h = F.gelu(h)
        # h: (B, E, H) @ w2[:E]: (E, H, O) -> all_out: (B, E, O)
        all_out = torch.einsum('beh,eho->beo', h, self.w2[:num_active]) + self.b2[:num_active]

        # Gather top-k expert outputs
        idx = top_k_indices.unsqueeze(-1).expand(-1, -1, self.output_dim)  # (B, k, O)
        selected = torch.gather(all_out, 1, idx)  # (B, k, O)

        # Weighted sum
        output = (top_k_probs.unsqueeze(-1) * selected).sum(dim=1)  # (B, O)

        # Load balancing loss
        aux_loss = self._load_balance_loss(router_probs, top_k_indices, num_active)

        return output, aux_loss

    def _load_balance_loss(self, router_probs, top_k_indices, num_active):
        """
        Switch Transformer style load balancing loss.
        loss = num_active * sum(f_i * p_i)
        """
        one_hot = F.one_hot(top_k_indices, num_classes=num_active)  # (B, k, num_active)
        f = one_hot.float().sum(dim=1).mean(dim=0)  # (num_active,)
        p = router_probs.mean(dim=0)  # (num_active,)
        return self.aux_loss_weight * num_active * (f * p).sum()

    def count_expert_params(self, num_active=None):
        """Count parameters for given number of active experts."""
        num_active = num_active or self.num_experts
        per_expert = (
            self.input_dim * self.hidden_dim + self.hidden_dim +
            self.hidden_dim * self.output_dim + self.output_dim
        )
        return per_expert * num_active
