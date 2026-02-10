"""
Structured Pruning

L1-norm based channel pruning: remove low-magnitude channels from a
trained model, then fine-tune to recover accuracy. Used as a baseline.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class StructuredPruner:
    """L1-norm based structured channel pruning with fine-tuning."""

    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.original_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def prune(self, prune_ratio=0.5, finetune_epochs=10, lr=0.001):
        """
        Prune the model and fine-tune.

        Args:
            prune_ratio: fraction of channels to prune (0.5 = remove half)
            finetune_epochs: number of fine-tuning epochs after pruning
            lr: fine-tuning learning rate

        Returns:
            pruned_model: pruned and fine-tuned model
            history: fine-tuning metrics
        """
        print(f"\nStructured Pruning -- ratio: {prune_ratio}")

        orig_params = sum(p.numel() for p in self.original_model.parameters())
        pruned_model = copy.deepcopy(self.original_model).to(self.device)

        self._apply_channel_pruning(pruned_model, prune_ratio)

        pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
        print(f"Params: {orig_params:,} -> {pruned_params:,} ({pruned_params/orig_params:.1%})")

        history = self._finetune(pruned_model, finetune_epochs, lr)

        return pruned_model, history

    def _apply_channel_pruning(self, model, prune_ratio):
        """Apply L1-norm based channel pruning to all conv layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data  # (out_c, in_c, kH, kW)
                l1_norms = weight.abs().sum(dim=(1, 2, 3))
                num_channels = weight.size(0)
                num_prune = int(num_channels * prune_ratio)

                if num_prune == 0:
                    continue

                # Zero out lowest-norm channels (soft pruning)
                _, prune_indices = torch.topk(l1_norms, num_prune, largest=False)
                mask = torch.ones(num_channels, device=weight.device)
                mask[prune_indices] = 0.0
                module.weight.data *= mask.view(-1, 1, 1, 1)

                if module.bias is not None:
                    module.bias.data *= mask

    def _finetune(self, model, epochs, lr):
        """Fine-tune the pruned model to recover accuracy."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            total_samples = 0

            for images, labels in tqdm(self.train_loader, desc=f"Finetune {epoch}", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

            val_acc = self._evaluate(model)
            avg_loss = total_loss / total_samples
            history["train_loss"].append(avg_loss)
            history["val_acc"].append(val_acc)

            print(f"  Finetune Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        return history

    @torch.no_grad()
    def _evaluate(self, model):
        model.eval()
        correct = 0
        total = 0
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = model(images)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        return correct / total
