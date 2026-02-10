"""
Visualization

Publication-ready plots for results:
1. Sub-model size vs accuracy (elastic vs pruning vs distillation)
2. Expert routing heatmap
3. Sub-model accuracy curves during training
4. Inference latency vs accuracy trade-off
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Non-interactive backend (Colab/server compatible)
matplotlib.use("Agg")

# Dark GitHub theme
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.size": 12,
    "axes.titlesize": 14,
    "figure.titlesize": 16,
})

# Color palette
COLORS = {
    "elastic": "#58a6ff",
    "pruning": "#f78166",
    "distillation": "#d2a8ff",
    "scratch": "#7ee787",
    "accent": "#ffa657",
}


class Visualizer:
    """Generates all benchmark and training plots."""

    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_accuracy_vs_params(self, benchmark_results, filename="accuracy_vs_params.png"):
        """
        Parameter count vs accuracy scatter plot.
        Each method is shown with a distinct color and marker.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        method_styles = {
            "elastic": {"color": COLORS["elastic"], "marker": "o", "s": 150, "label": "Elastic (ours)"},
            "pruning": {"color": COLORS["pruning"], "marker": "s", "s": 120, "label": "Pruning"},
            "distillation": {"color": COLORS["distillation"], "marker": "^", "s": 120, "label": "Distillation"},
            "scratch": {"color": COLORS["scratch"], "marker": "D", "s": 120, "label": "Train from scratch"},
        }

        for result in benchmark_results:
            method = result.get("method", "elastic")
            style = method_styles.get(method, method_styles["elastic"])

            ax.scatter(
                result["params"],
                result["accuracy"] * 100,
                color=style["color"],
                marker=style["marker"],
                s=style["s"],
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
            )
            # Label
            offset = (10, 5) if result["accuracy"] > 0.5 else (10, -15)
            ax.annotate(
                result["name"],
                (result["params"], result["accuracy"] * 100),
                textcoords="offset points",
                xytext=offset,
                fontsize=9,
                color=style["color"],
            )

        # Legend (unique entries)
        seen = set()
        handles = []
        for result in benchmark_results:
            method = result.get("method", "elastic")
            if method not in seen:
                style = method_styles.get(method, method_styles["elastic"])
                handles.append(ax.scatter([], [], color=style["color"], marker=style["marker"],
                                         s=style["s"], label=style["label"], edgecolors="white"))
                seen.add(method)
        ax.legend(handles=handles, loc="lower right", framealpha=0.8,
                  facecolor="#161b22", edgecolor="#30363d")

        ax.set_xlabel("Parameters")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Elastic Training: One Training Run -> Multiple Models")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return path

    def plot_training_curves(self, elastic_history, filename="training_curves.png"):
        """
        Sub-model accuracy curves during training.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        epochs = range(1, len(elastic_history["train_loss"]) + 1)

        # Left: Training loss
        ax1.plot(epochs, elastic_history["train_loss"], color=COLORS["elastic"], linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)

        # Right: Sub-model accuracies
        if "val_acc_large" in elastic_history:
            ax2.plot(epochs, [a * 100 for a in elastic_history["val_acc_large"]],
                     color="#58a6ff", linewidth=2, label="Large (6L/8E/top-2)")
            ax2.plot(epochs, [a * 100 for a in elastic_history["val_acc_medium"]],
                     color="#ffa657", linewidth=2, label="Medium (4L/6E/top-2)")
            ax2.plot(epochs, [a * 100 for a in elastic_history["val_acc_small"]],
                     color="#7ee787", linewidth=2, label="Small (3L/4E/top-1)")
        else:
            ax2.plot(epochs, [a * 100 for a in elastic_history["val_acc"]],
                     color=COLORS["elastic"], linewidth=2, label="Validation Acc")

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Sub-Model Accuracy During Training")
        ax2.legend(framealpha=0.8, facecolor="#161b22", edgecolor="#30363d")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return path

    def plot_routing_heatmap(self, router_info, filename="routing_heatmap.png"):
        """
        Expert routing heatmap: shows how frequently each expert is activated.
        router_info: list of routing data returned from a forward pass.
        """
        if not router_info:
            print("No routing info to plot.")
            return None

        fig, axes = plt.subplots(1, len(router_info), figsize=(4 * len(router_info), 6))
        if len(router_info) == 1:
            axes = [axes]

        for idx, info in enumerate(router_info):
            probs = info["probs"]  # (batch, num_experts)
            if probs is None:
                continue

            probs_np = probs.cpu().numpy()
            avg_probs = probs_np.mean(axis=0)  # Average probability per expert

            # Heatmap (single row reshaped for display)
            num_experts = len(avg_probs)
            heatmap_data = avg_probs.reshape(1, -1)

            sns.heatmap(
                heatmap_data,
                ax=axes[idx],
                cmap="YlOrRd",
                annot=True,
                fmt=".3f",
                xticklabels=[f"E{i}" for i in range(num_experts)],
                yticklabels=["Prob"],
                cbar=idx == len(router_info) - 1,
                vmin=0,
                vmax=max(0.3, avg_probs.max()),
            )
            axes[idx].set_title(f"Block {info['block']}")

        plt.suptitle("Expert Routing Probabilities", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return path

    def plot_latency_vs_accuracy(self, benchmark_results, filename="latency_vs_accuracy.png"):
        """
        Inference latency vs accuracy trade-off scatter plot.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        method_colors = {
            "elastic": COLORS["elastic"],
            "pruning": COLORS["pruning"],
            "distillation": COLORS["distillation"],
            "scratch": COLORS["scratch"],
            "baseline": COLORS["accent"],
        }

        for result in benchmark_results:
            method = result.get("method", "elastic")
            color = method_colors.get(method, COLORS["accent"])

            ax.scatter(
                result["latency_ms"],
                result["accuracy"] * 100,
                color=color,
                s=150,
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
            )
            ax.annotate(
                result["name"],
                (result["latency_ms"], result["accuracy"] * 100),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=9,
                color=color,
            )

        ax.set_xlabel("Inference Latency (ms)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs Inference Speed Trade-off")
        ax.grid(True, alpha=0.3)

        # Draw Pareto frontier between elastic points
        elastic_results = [r for r in benchmark_results if r.get("method") == "elastic"]
        if len(elastic_results) >= 2:
            elastic_sorted = sorted(elastic_results, key=lambda r: r["latency_ms"])
            latencies = [r["latency_ms"] for r in elastic_sorted]
            accuracies = [r["accuracy"] * 100 for r in elastic_sorted]
            ax.plot(latencies, accuracies, "--", color=COLORS["elastic"], alpha=0.5, linewidth=1.5)

        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return path

    def plot_method_comparison_bar(self, benchmark_results, filename="method_comparison.png"):
        """
        Bar chart comparing accuracy across all methods.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        names = [r["name"] for r in benchmark_results]
        accuracies = [r["accuracy"] * 100 for r in benchmark_results]
        methods = [r.get("method", "elastic") for r in benchmark_results]

        colors = [
            COLORS.get(m, COLORS["accent"]) for m in methods
        ]

        bars = ax.bar(range(len(names)), accuracies, color=colors, edgecolor="white", linewidth=0.5)

        # Display values above bars
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#c9d1d9",
            )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Method Comparison: Elastic Training vs Baselines")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        return path

    def generate_all_plots(self, elastic_history, benchmark_results, router_info=None):
        """Generate all plots at once."""
        paths = []
        paths.append(self.plot_training_curves(elastic_history))
        paths.append(self.plot_accuracy_vs_params(benchmark_results))
        paths.append(self.plot_latency_vs_accuracy(benchmark_results))
        paths.append(self.plot_method_comparison_bar(benchmark_results))
        if router_info:
            paths.append(self.plot_routing_heatmap(router_info))
        print(f"\nAll plots saved to '{self.save_dir}/'.")
        return paths
