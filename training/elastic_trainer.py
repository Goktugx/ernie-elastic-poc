"""
Elastic Training Loop

Features:
- Sandwich rule (largest + smallest + random config per step)
- Progressive elastic: start with large configs, gradually add smaller ones
- Loss weighting: larger sub-models get higher weight
- Warmup + cosine annealing scheduler
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ElasticTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device="cuda",
        use_sandwich_rule=True,
        use_progressive=True,
        loss_weights=None,
        log_fn=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_sandwich_rule = use_sandwich_rule
        self.use_progressive = use_progressive
        self.log_fn = log_fn

        # Loss weights: [largest, smallest, random]
        # Larger sub-model gets more weight to boost elastic-large accuracy
        self.loss_weights = loss_weights or [0.5, 0.2, 0.3]

        self.history = {
            "train_loss": [],
            "val_acc": [],
            "val_acc_large": [],
            "val_acc_medium": [],
            "val_acc_small": [],
            "configs": [],
            "epoch_times": [],
            "lr": [],
        }

        # For progressive elastic
        self._current_epoch = 0
        self._total_epochs = 1

    def _get_progressive_choices(self, epoch, total_epochs):
        """
        Progressive elastic: gradually expand the search space.
        Phase 1 (0-33%): only large configs
        Phase 2 (33-66%): large + medium configs
        Phase 3 (66-100%): all configs
        """
        if not self.use_progressive:
            return (
                self.model.depth_choices,
                self.model.width_choices,
                self.model.sparsity_choices,
            )

        progress = epoch / total_epochs
        all_d = sorted(self.model.depth_choices)
        all_w = sorted(self.model.width_choices)
        all_s = sorted(self.model.sparsity_choices)

        if progress < 0.33:
            # Phase 1: only largest configs
            depth = all_d[-1:]          # [6]
            width = all_w[-1:]          # [8] or [16]
            sparsity = all_s[-1:]       # [3]
        elif progress < 0.66:
            # Phase 2: top half
            mid_d = max(1, len(all_d) // 2)
            mid_w = max(1, len(all_w) // 2)
            mid_s = max(1, len(all_s) // 2)
            depth = all_d[mid_d:]
            width = all_w[mid_w:]
            sparsity = all_s[mid_s:]
        else:
            # Phase 3: all configs
            depth = all_d
            width = all_w
            sparsity = all_s

        return depth, width, sparsity

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        total_samples = 0
        start_time = time.time()
        self._current_epoch = epoch
        self._total_epochs = total_epochs

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_sandwich_rule:
                loss = self._sandwich_step(images, labels, epoch, total_epochs)
            else:
                loss = self._random_step(images, labels, epoch, total_epochs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{total_loss / total_samples:.4f}", "lr": f"{current_lr:.6f}"})

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples

        self.history["train_loss"].append(avg_loss)
        self.history["epoch_times"].append(epoch_time)
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

        return avg_loss, epoch_time

    def _sandwich_step(self, images, labels, epoch, total_epochs):
        """Sandwich rule with progressive elastic and loss weighting."""
        depth_c, width_c, sparsity_c = self._get_progressive_choices(epoch, total_epochs)

        import random
        largest = {
            "depth": max(depth_c),
            "width": max(width_c),
            "sparsity": max(sparsity_c),
        }
        smallest = {
            "depth": min(depth_c),
            "width": min(width_c),
            "sparsity": min(sparsity_c),
        }
        random_config = {
            "depth": random.choice(depth_c),
            "width": random.choice(width_c),
            "sparsity": random.choice(sparsity_c),
        }

        configs = [largest, smallest, random_config]
        weights = self.loss_weights
        total_loss = 0

        for config, w in zip(configs, weights):
            logits, aux_loss, _ = self.model(images, elastic_config=config)
            ce_loss = F.cross_entropy(logits, labels)
            total_loss += w * (ce_loss + aux_loss)

        return total_loss

    def _random_step(self, images, labels, epoch, total_epochs):
        """Single random config with progressive elastic."""
        import random
        depth_c, width_c, sparsity_c = self._get_progressive_choices(epoch, total_epochs)
        config = {
            "depth": random.choice(depth_c),
            "width": random.choice(width_c),
            "sparsity": random.choice(sparsity_c),
        }
        logits, aux_loss, _ = self.model(images, elastic_config=config)
        ce_loss = F.cross_entropy(logits, labels)
        return ce_loss + aux_loss

    @torch.no_grad()
    def evaluate(self, elastic_config=None, data_loader=None):
        self.model.eval()
        loader = data_loader or self.val_loader
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits, _, _ = self.model(images, elastic_config=elastic_config)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        return correct / total

    def evaluate_all_submodels(self):
        results = {}
        for size in ["large", "medium", "small"]:
            config = self.model.get_submodel_config(size)
            acc = self.evaluate(elastic_config=config)
            results[size] = acc
        return results

    def train(self, num_epochs):
        print(f"\nElastic Training starting -- {num_epochs} epochs")
        print(f"Sandwich rule: {self.use_sandwich_rule}")
        print(f"Progressive elastic: {self.use_progressive}")
        print(f"Loss weights (large/small/random): {self.loss_weights}")
        print(f"Depth choices: {self.model.depth_choices}")
        print(f"Width choices: {self.model.width_choices}")
        print(f"Sparsity choices: {self.model.sparsity_choices}")
        print(f"Num experts: {self.model.num_experts}")
        print("-" * 60)

        for epoch in range(1, num_epochs + 1):
            train_loss, epoch_time = self.train_epoch(epoch, num_epochs)

            sub_results = self.evaluate_all_submodels()

            self.history["val_acc_large"].append(sub_results["large"])
            self.history["val_acc_medium"].append(sub_results["medium"])
            self.history["val_acc_small"].append(sub_results["small"])
            self.history["val_acc"].append(sub_results["large"])

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                f"Large: {sub_results['large']:.4f} | "
                f"Medium: {sub_results['medium']:.4f} | "
                f"Small: {sub_results['small']:.4f} | "
                f"LR: {lr:.6f} | Time: {epoch_time:.1f}s"
            )

            if self.log_fn:
                self.log_fn({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_acc_large": sub_results["large"],
                    "val_acc_medium": sub_results["medium"],
                    "val_acc_small": sub_results["small"],
                    "lr": lr,
                    "epoch_time": epoch_time,
                })

        return self.history


class StandardTrainer:
    """Standard training loop for baseline models."""

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.history = {"train_loss": [], "val_acc": [], "epoch_times": []}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_samples = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            pbar.set_postfix({"loss": f"{total_loss / total_samples:.4f}"})

        epoch_time = time.time() - start_time
        if self.scheduler:
            self.scheduler.step()
        self.history["train_loss"].append(total_loss / total_samples)
        self.history["epoch_times"].append(epoch_time)
        return total_loss / total_samples, epoch_time

    @torch.no_grad()
    def evaluate(self, data_loader=None):
        self.model.eval()
        loader = data_loader or self.val_loader
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        return correct / total

    def train(self, num_epochs):
        print(f"\nStandard Training -- {num_epochs} epochs")
        print("-" * 60)
        for epoch in range(1, num_epochs + 1):
            train_loss, epoch_time = self.train_epoch(epoch)
            val_acc = self.evaluate()
            self.history["val_acc"].append(val_acc)
            print(
                f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s"
            )
        return self.history
