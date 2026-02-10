# ERNIE Elastic PoC

Small-scale proof of concept for **ERNIE 5.0's elastic training** idea: train a single Mixture-of-Experts (MoE) model and extract different-sized sub-models from one checkpoint -- no retraining needed.

## The Idea

Instead of training separate small/medium/large models, train **one elastic model** that supports three flexibility axes:

| Axis | What it controls | Range |
|------|-----------------|-------|
| **Elastic Depth** | Active MoE blocks per step | 3-6 layers |
| **Elastic Width** | Active experts per step | 4-16 experts |
| **Elastic Sparsity** | Top-k routing per step | top-1 to top-3 |

After training, extract sub-models by simply changing the config:

```
Large  = {depth: 6, width: 16, sparsity: 2}  -- full model
Medium = {depth: 4, width: 12, sparsity: 2}  -- 60% params
Small  = {depth: 3, width: 8,  sparsity: 1}  -- 35% params
```

## Architecture

```
Input (3x32x32)
    |
Conv Feature Extractor (fixed)
    |
MoE Block x6 (elastic depth/width/sparsity)
  |- LayerNorm + MoE Layer + Residual
  |- 16 experts with top-k routing
  |- Load balancing loss (Switch Transformer style)
    |
Classification Head
    |
Output (10 classes)
```

## Training Techniques

- **Sandwich Rule** (from OFA): train largest + smallest + random config each step
- **Progressive Elastic**: gradually expand config space (large only -> top half -> all)
- **Loss Weighting**: [0.5, 0.2, 0.3] for [largest, smallest, random]
- **Warmup + Cosine Annealing**: per-step LR with 5% linear warmup
- **Data Augmentation**: RandomCrop, HorizontalFlip, ColorJitter

## Baselines

We compare elastic sub-models against four classical methods:

1. **Large CNN** -- trained from scratch
2. **Structured Pruning** -- L1-norm channel pruning + fine-tuning
3. **Knowledge Distillation** -- Hinton KD (temperature=4.0, alpha=0.7)
4. **Small CNN** -- trained from scratch

## Quick Start

```bash
# Setup
pip install torch torchvision tqdm seaborn matplotlib numpy

# Full pipeline (elastic + baselines + benchmark + plots)
python main.py --epochs 120 --num_experts 16

# Or run stages separately
python main.py --stage elastic --epochs 120
python main.py --stage baselines --epochs 120
python main.py --stage benchmark
python main.py --stage visualize
```

### Google Colab

Open `colab.ipynb` or clone and run:

```python
!git clone https://github.com/YOUR_USERNAME/ernie-elastic-poc.git
%cd ernie-elastic-poc
!pip install torch torchvision tqdm seaborn
```

## Project Structure

```
models/
  moe_layer.py          # Expert, Router, MoELayer with load balancing
  elastic_moe_model.py  # ElasticMoEModel with 3 elasticity axes
  baseline_model.py     # BaselineCNN (small/medium/large)

training/
  elastic_trainer.py    # ElasticTrainer with sandwich rule + progressive
  pruning.py            # Structured L1-norm pruning
  distillation.py       # Hinton knowledge distillation

evaluation/
  extract_submodel.py   # Sub-model extraction from elastic checkpoint
  benchmark.py          # Accuracy + latency benchmarking

visualization/
  plots.py              # Training curves, comparisons, routing heatmaps

main.py                 # Full pipeline orchestrator
notebook.ipynb          # Interactive notebook (VSCode / Jupyter)
```

## Outputs

- `checkpoints/` -- model weights
- `results/` -- benchmark results (JSON)
- `plots/` -- visualizations (PNG)

## References

- [ERNIE 5.0](https://arxiv.org/abs/2503.04648) -- Baidu's elastic training for MoE
- [Once-for-All (OFA)](https://arxiv.org/abs/1908.09791) -- sandwich rule, progressive shrinking
- [Switch Transformer](https://arxiv.org/abs/2101.03961) -- load balancing loss for MoE

## Dataset

CIFAR-10: 60k images (50k train / 10k test), 32x32 RGB, 10 classes. Chosen for fast iteration -- the elastic training concept scales to larger datasets and models.
