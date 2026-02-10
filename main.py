"""
Elastic Training -- Main Script

Small-scale proof of concept for ERNIE 5.0's elastic training idea:
extract different-sized sub-models from a single training run
and compare them against classical methods (pruning, distillation).

Usage:
    python main.py                          # Full pipeline
    python main.py --stage elastic          # Elastic training only
    python main.py --stage baselines        # Baseline training only
    python main.py --stage benchmark        # Benchmark only
    python main.py --stage visualize        # Visualization only
    python main.py --epochs 30 --batch_size 128
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models import ElasticMoEModel, BaselineCNN
from training.elastic_trainer import ElasticTrainer, StandardTrainer, get_warmup_cosine_scheduler
from training.pruning import StructuredPruner
from training.distillation import DistillationTrainer
from evaluation.extract_submodel import SubModelExtractor
from evaluation.benchmark import Benchmarker
from visualization.plots import Visualizer


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_data_loaders(batch_size=128, num_workers=2):
    """Create CIFAR-10 data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader


def stage_elastic_training(args, train_loader, test_loader, device):
    """Stage 1-2: Train elastic MoE model."""
    print("\n" + "=" * 60)
    print("ELASTIC TRAINING")
    print("=" * 60)

    model = ElasticMoEModel(
        num_classes=10,
        num_blocks=6,
        embed_dim=128,
        moe_hidden_dim=256,
        num_experts=args.num_experts,
        default_top_k=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Num experts: {args.num_experts}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup
    scheduler = get_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps)

    trainer = ElasticTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_sandwich_rule=True,
        use_progressive=True,
        loss_weights=[0.5, 0.2, 0.3],
    )

    start_time = time.time()
    history = trainer.train(num_epochs=args.epochs)
    total_time = time.time() - start_time

    history["total_time"] = total_time
    print(f"\nTotal training time: {total_time:.1f}s")

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/elastic_moe.pt")
    with open("checkpoints/elastic_history.json", "w") as f:
        json.dump({k: v for k, v in history.items() if k != "configs"}, f, indent=2)

    # Sub-model extraction report
    extractor = SubModelExtractor(model)
    extractor.extract_all_presets()

    return model, history


def stage_baselines(args, train_loader, test_loader, device, teacher_model=None):
    """Stage 4: Train baseline models."""
    print("\n" + "=" * 60)
    print("BASELINE TRAINING")
    print("=" * 60)

    results = {}
    os.makedirs("checkpoints", exist_ok=True)

    # --- Baseline 1: Large CNN (teacher for pruning) ---
    print("\n--- Baseline 1: Large CNN ---")
    large_cnn = BaselineCNN(num_classes=10, size="large")
    optimizer = optim.AdamW(large_cnn.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = StandardTrainer(large_cnn, train_loader, test_loader, optimizer, scheduler, device)
    large_history = trainer.train(args.epochs)
    results["large_cnn"] = {"model": large_cnn, "history": large_history}

    # --- Baseline 2: Pruning (large CNN -> prune) ---
    print("\n--- Baseline 2: Pruning ---")
    pruner = StructuredPruner(large_cnn, train_loader, test_loader, device)
    pruned_model, prune_history = pruner.prune(
        prune_ratio=0.5, finetune_epochs=max(5, args.epochs // 4), lr=args.lr * 0.1
    )
    results["pruned"] = {"model": pruned_model, "history": prune_history}

    # --- Baseline 3: Distillation (large CNN -> small CNN) ---
    print("\n--- Baseline 3: Knowledge Distillation ---")
    # Teacher: use elastic model if available, otherwise large CNN
    if teacher_model is not None:
        teacher = teacher_model
    else:
        teacher = large_cnn

    student = BaselineCNN(num_classes=10, size="small")
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    distiller = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        device=device,
        temperature=4.0,
        alpha=0.7,
    )
    distill_history = distiller.train(args.epochs)
    results["distilled"] = {"model": student, "history": distill_history}

    # --- Baseline 4: Small CNN from scratch ---
    print("\n--- Baseline 4: Small CNN (from scratch) ---")
    small_cnn = BaselineCNN(num_classes=10, size="small")
    optimizer = optim.AdamW(small_cnn.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = StandardTrainer(small_cnn, train_loader, test_loader, optimizer, scheduler, device)
    small_history = trainer.train(args.epochs)
    results["small_scratch"] = {"model": small_cnn, "history": small_history}

    # Save checkpoints
    for name, data in results.items():
        torch.save(data["model"].state_dict(), f"checkpoints/{name}.pt")

    return results


def stage_benchmark(elastic_model, baseline_results, test_loader, device):
    """Stage 4 continued: Benchmark all models."""
    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)

    benchmarker = Benchmarker(test_loader, device=device, num_warmup=5, num_runs=50)

    baseline_models = {}
    for name, data in baseline_results.items():
        baseline_models[name] = data["model"]

    results = benchmarker.run_full_comparison(elastic_model, baseline_models)

    # Save results
    os.makedirs("results", exist_ok=True)
    serializable = []
    for r in results:
        s = {k: v for k, v in r.items() if k != "config"}
        if "config" in r and r["config"] is not None:
            s["config"] = r["config"]
        serializable.append(s)

    with open("results/benchmark_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    return results


def stage_visualize(elastic_history, benchmark_results, elastic_model, test_loader, device):
    """Stage 5: Generate plots."""
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    viz = Visualizer(save_dir="plots")

    # Run a forward pass to get router info
    elastic_model.eval()
    elastic_model.to(device)
    router_info = None
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        config = elastic_model.get_submodel_config("large")
        _, _, router_info = elastic_model(images, elastic_config=config)

    viz.generate_all_plots(elastic_history, benchmark_results, router_info)


def main():
    parser = argparse.ArgumentParser(description="Elastic Training PoC")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of MoE experts (8 or 16)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "elastic", "baselines", "benchmark", "visualize"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")

    # Data
    train_loader, test_loader = get_data_loaders(args.batch_size, args.num_workers)
    print(f"CIFAR-10: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    if args.stage in ("all", "elastic"):
        elastic_model, elastic_history = stage_elastic_training(
            args, train_loader, test_loader, device
        )
    else:
        # Load from checkpoint
        elastic_model = ElasticMoEModel(num_experts=args.num_experts)
        elastic_model.load_state_dict(torch.load("checkpoints/elastic_moe.pt", weights_only=True))
        elastic_model.to(device)
        with open("checkpoints/elastic_history.json") as f:
            elastic_history = json.load(f)

    if args.stage in ("all", "baselines"):
        baseline_results = stage_baselines(
            args, train_loader, test_loader, device, teacher_model=elastic_model
        )
    else:
        baseline_results = {}
        for name, size in [("large_cnn", "large"), ("small_scratch", "small")]:
            path = f"checkpoints/{name}.pt"
            if os.path.exists(path):
                m = BaselineCNN(num_classes=10, size=size)
                m.load_state_dict(torch.load(path, weights_only=True))
                m.to(device)
                baseline_results[name] = {"model": m, "history": {}}

    if args.stage in ("all", "benchmark"):
        benchmark_results = stage_benchmark(
            elastic_model, baseline_results, test_loader, device
        )
    else:
        if os.path.exists("results/benchmark_results.json"):
            with open("results/benchmark_results.json") as f:
                benchmark_results = json.load(f)
        else:
            benchmark_results = []

    if args.stage in ("all", "visualize"):
        stage_visualize(elastic_history, benchmark_results, elastic_model, test_loader, device)

    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print("Outputs:")
    print("  checkpoints/  - model weights")
    print("  results/      - benchmark results (JSON)")
    print("  plots/        - visualizations (PNG)")


if __name__ == "__main__":
    main()
