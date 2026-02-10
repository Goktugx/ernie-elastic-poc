"""
Benchmark

Accuracy and inference latency measurement.
Compares elastic sub-models against baseline methods.
"""

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


class Benchmarker:
    """Measures model accuracy and inference latency."""

    def __init__(self, val_loader, device="cuda", num_warmup=10, num_runs=100):
        self.val_loader = val_loader
        self.device = device
        self.num_warmup = num_warmup
        self.num_runs = num_runs

    @torch.no_grad()
    def measure_accuracy(self, model, elastic_config=None):
        """Measure validation accuracy."""
        model.eval()
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            if elastic_config is not None:
                logits, _, _ = model(images, elastic_config=elastic_config)
            else:
                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output

            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        return correct / total

    @torch.no_grad()
    def measure_inference_time(self, model, elastic_config=None, batch_size=64):
        """
        Measure average inference latency per batch (ms).
        Uses GPU synchronization for accurate timing.
        """
        model.eval()
        dummy_input = torch.randn(batch_size, 3, 32, 32, device=self.device)

        # Warmup
        for _ in range(self.num_warmup):
            if elastic_config is not None:
                model(dummy_input, elastic_config=elastic_config)
            else:
                model(dummy_input)

        # Measurement
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(self.num_runs):
            if elastic_config is not None:
                model(dummy_input, elastic_config=elastic_config)
            else:
                model(dummy_input)

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / self.num_runs * 1000  # ms
        return elapsed

    def count_params(self, model, elastic_config=None):
        """Count model parameters."""
        if elastic_config is not None and hasattr(model, "count_active_params"):
            return model.count_active_params(elastic_config)
        return sum(p.numel() for p in model.parameters())

    def benchmark_model(self, model, name, elastic_config=None):
        """Benchmark a single model and return results."""
        acc = self.measure_accuracy(model, elastic_config)
        latency = self.measure_inference_time(model, elastic_config)
        params = self.count_params(model, elastic_config)

        result = {
            "name": name,
            "accuracy": acc,
            "latency_ms": latency,
            "params": params,
            "config": elastic_config,
        }

        print(f"  {name:30s} | Acc: {acc:.4f} | Latency: {latency:.2f}ms | Params: {params:,}")
        return result

    def run_full_comparison(self, elastic_model, baseline_models):
        """
        Full comparison: elastic sub-models + baseline models.

        Args:
            elastic_model: trained ElasticMoEModel
            baseline_models: dict {"name": model}

        Returns:
            results: list of all benchmark results
        """
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        results = []

        # Elastic sub-models
        print("\n--- Elastic Sub-Models (from single training run) ---")
        for size in ["large", "medium", "small"]:
            config = elastic_model.get_submodel_config(size)
            result = self.benchmark_model(
                elastic_model, f"Elastic-{size}", elastic_config=config
            )
            result["method"] = "elastic"
            result["size"] = size
            results.append(result)

        # Baseline models
        print("\n--- Baseline Models ---")
        for name, model in baseline_models.items():
            result = self.benchmark_model(model, name)
            result["method"] = "baseline"
            results.append(result)

        print("=" * 80)
        return results
