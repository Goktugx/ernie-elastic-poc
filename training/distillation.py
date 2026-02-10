"""
Knowledge Distillation

Train a small student model using soft outputs from a large teacher model.
Hinton et al. (2015) style KD used as a baseline comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DistillationTrainer:
    """
    Hinton et al. (2015) knowledge distillation.

    Loss = alpha * KL(soft_student, soft_teacher) + (1-alpha) * CE(student, labels)

    - Temperature (T): softens teacher outputs for richer information
    - Alpha: balances soft vs hard loss
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        device="cuda",
        temperature=4.0,
        alpha=0.7,
    ):
        self.teacher = teacher_model.to(device).eval()
        self.student = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.history = {"train_loss": [], "val_acc": []}

        # Freeze teacher weights
        for p in self.teacher.parameters():
            p.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """KD loss = alpha * KL_div(soft) + (1 - alpha) * CE(hard)"""
        T = self.temperature

        # Soft targets (temperature scaled)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

        # Hard targets (standard cross-entropy)
        ce_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss

    def train_epoch(self, epoch):
        self.student.train()
        total_loss = 0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc=f"Distill Epoch {epoch}")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Teacher inference (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                # Handle MoE models that return tuples
                if isinstance(teacher_logits, tuple):
                    teacher_logits = teacher_logits[0]

            # Student forward
            student_logits = self.student(images)
            if isinstance(student_logits, tuple):
                student_logits = student_logits[0]

            loss = self.distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            pbar.set_postfix({"loss": f"{total_loss / total_samples:.4f}"})

        return total_loss / total_samples

    @torch.no_grad()
    def evaluate(self):
        self.student.eval()
        correct = 0
        total = 0
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.student(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        return correct / total

    def train(self, num_epochs):
        print(f"\nKnowledge Distillation -- {num_epochs} epochs")
        print(f"Temperature: {self.temperature} | Alpha: {self.alpha}")
        print("-" * 60)

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_acc = self.evaluate()

            self.history["train_loss"].append(train_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        return self.history
