# Umbilic Multinomial Logistic Regression

Official code for **Umbilic Multinomial Logistic Regression** (NeurIPS 2026).

Zihan Su (Harvard University), Nicu Sebe (University of Trento), Bernhard Schölkopf (MPI for Intelligent Systems, Tübingen), Ziheng Chen (University of Trento; MPI for Intelligent Systems, Tübingen)

## Overview

A Euclidean MLR logit is a scaled signed distance to a class hyperplane. In
Euclidean space, the complete connected totally umbilic hypersurfaces are
hyperplanes and spheres, so the same signed-distance construction gives a
second class coordinate: the signed distance to a class sphere with a learned
center and radius. UMLR places both coordinates, with a first-order or a
signed-square link, in one family:

| Head | Class-k logit | Code |
|---|---|---|
| UMLR-H (MLR) | ⟨a_k, x⟩ + b_k | `HyperplaneHead(score_mode="linear")` |
| UMLR²-H | α_k · sign(z) z², with z = ⟨a_k, x⟩ + b_k | `HyperplaneHead(score_mode="signed_square")` |
| UMLR-S | α_k · (r_k − ‖x − c_k‖) | `SphereUMLRHead(boundary_mode="distance")` |
| UMLR²-S | α_k · sign(z) z², with z = r_k − ‖x − c_k‖ | `SphereUMLRHead(boundary_mode="signed_square")` |

With r_k = 0 and α_k = 1, UMLR²-S is the squared-distance prototype
score −‖x − c_k‖². Because the family separates the class geometry, the score
link, and the radius, it supports controlled comparisons of hyperplane and
spherical heads. The repository also contains CosFace/ArcFace, a learned
prototype head, and the exploratory hyperplane–sphere mixed heads.

## Installation

```bash
git clone https://github.com/NickyoyoSu/Umbilic-Multinomial-Logistic-Regression.git
cd Umbilic-Multinomial-Logistic-Regression
pip install -e ".[episodic,figures,dev]"
pytest
```

The experiments were run with Python 3.11 and PyTorch 2 on NVIDIA A100 GPUs
(one GPU per run; four GPUs for caching ImageNet-1K features). With
PyTorch < 2.3, install `numpy<2`.

## Using the heads

```python
import torch
from umlr import BoundaryClassifier, HyperplaneHead, SphereUMLRHead

feature_dim, num_classes = 768, 100
head = SphereUMLRHead(feature_dim, num_classes, boundary_mode="signed_square", alpha_init_effective=1.0)
# Optional: start from class feature means and a within-class distance quantile.
# head.set_centers_(class_means); head.set_radius_values_(class_radii)

backbone = torch.nn.Identity()  # any feature extractor
model = BoundaryClassifier(backbone, head, normalize_features=True)
logits = model(torch.randn(8, feature_dim))
```

## Data

```bash
python scripts/download_datasets.py --data-root data
```

downloads CIFAR-100, Flowers102, TinyImageNet, Oxford-IIIT Pet, and Food-101
through torchvision (TinyImageNet from the CS231n mirror). Notes on splits:

- CIFAR-100, Oxford-IIIT Pet: official train/test splits.
- Flowers102: the official `train` split is used for training and `val` for
  evaluation. The `train` split has 10 images per class; a request for more
  labeled images per class than are available uses all images of the class.
- TinyImageNet: official training folders; validation images are sorted into
  class folders and used for evaluation.
- Food-101: a seeded 10% hold-out of the official training split is used for
  evaluation.
- Low-shot subsets are drawn per class from the training split with the run
  seed.
- ImageNet-1K (Experiment 4) must be obtained from
  [image-net.org](https://image-net.org) and arranged as `train/` and `val/`
  in ImageFolder layout, with validation images sorted into class folders.
- Few-shot episodic datasets (Experiment 3) are fetched on first use:
  miniImageNet, CIFAR-FS, and FC100 from the Hugging Face hub
  (`GATE-engine/*`), CUB and tieredImageNet from
  `pancake/few_shot_datasets`.

## Reproducing the paper

Each script in `experiments/` runs one experiment sequentially on one GPU and
writes runs to `outputs/<tag>/` and summaries to `outputs/reports/`. All scripts
accept `DATA_ROOT`, `OUTPUT_ROOT`, `SEEDS` (default `0 1 2 3 4`), and
`DRY_RUN=1` (print the commands only); the per-script filters
(`DATASETS`, `BACKBONES`, `SHOTS`, ...) are listed in each file. Each line of
the dry-run output is an independent job and can be submitted to a cluster.

| Paper | Script | Summary |
|---|---|---|
| Table 4 (Experiment 1), Table 6 (Experiment 2, UMLR²-S versus its zero-radius limit) | `experiments/exp1_head_swap.sh` | `reports/exp1_head_swap.md` |
| Table 5 (parameter-matched shot curve) | `experiments/exp1_shot_curve.sh` | `reports/shot_curve/` |
| Experiment 2, radius-only control | `experiments/exp2_radius_control.sh` | `reports/radius_control/` |
| Table 7 (Experiment 3, validation-selected episodic) and the appendix four-endpoint table | `experiments/exp3_episodic.sh` (`FOUR_ENDPOINT_TEST=1`) | `reports/episodic_*.md` |
| Table 8 (Experiment 4, ImageNet-1K) | `experiments/exp4_imagenet.sh` | `reports/imagenet.md` |
| Appendix: radius and scale sensitivity, frozen vs. fine-tuned, full-data and long-tail controls, large-K benchmark | `experiments/appendix_controls.sh` | `reports/{sensitivity,frozen_vs_finetuned,full_longtail}.md`, `reports/large_k.json` |
| Appendix: exploratory mixed heads and gate audit | `experiments/appendix_mixed.sh` | `reports/mixed*.md` |
| Figures 1 and 2 | `scripts/figures/make_geometry_schematic.py`, `scripts/figures/sample_features.py` + `scripts/figures/plot_tsne_triptych.py` | `figures/` |

For example, to run one seed of the low-shot head swap on CIFAR-100 with ViT-B/16:

```bash
SEEDS=0 DATASETS=cifar100 BACKBONES=vit_b_16 RUN_FULL_DATA=0 bash experiments/exp1_head_swap.sh
```

### Protocol details

- **Entry points.** `scripts/train.py` trains a head on a torchvision backbone
  (ViT-B/16 or ConvNeXt-Tiny with ImageNet weights). `scripts/run_episodic.py`
  adapts heads on few-shot episodes of cached features, and
  `scripts/select_episodic_endpoint.py` locks one endpoint from validation
  results before any test run (protocol: `experiments/episodic_protocol.json`).
  `scripts/extract_imagenet_features.py` and `scripts/train_imagenet_head.py`
  implement the ImageNet-1K head retraining.
- **Frozen features.** Low-shot runs freeze the backbone and pass the
  (augmented) training images through it every epoch. Features are centered by
  the training-set feature mean and L2-normalized (CL2N) before the head.
- **Initialization.** Spherical heads start from class feature means and a
  within-class distance quantile (`--feature-stat-radius-quantile`, 0.9 in the
  head-swap suite and 0.8 in the parameter-matched recipe).
- **Zero-radius limit.** The prototype column of Table 6 and the r = 0 arm of
  the radius-only control use the same head with every radius fixed at zero
  and the same class-mean centers (`--radius-scope fixed --fixed-radius 0
  --feature-stat-centers-only`), so they differ from UMLR²-S only in the
  radius.
- **Recipes.** The recipes of every table are defined in
  `experiments/common.sh` (head-swap, full-data fine-tuning, mixed-head, and
  parameter-matched recipes).
- **Metrics.** Every `scripts/train.py` epoch is evaluated on the evaluation
  split. `metrics.json` records `best_val_accuracy` (maximum over epochs) and
  `final_val_accuracy` (last epoch), and the summaries report both. The
  ImageNet-1K table reports the final epoch.
- **Episodic adaptation.** All four endpoints are adapted for 100 AdamW steps
  (learning rate 0.01, weight decay 1e-4, gradient clipping at 10, initial
  scale 10); spherical endpoints add a weak radius prior
  `1e-3 * mean_k (r_k - r_0)^2` toward the median base-class q90 radius `r_0`.

## Repository layout

```
src/umlr/            heads, classifier wrapper, backbones, data, training loop, episodic helpers
scripts/             training and evaluation entry points
scripts/analysis/    paired statistics and table summaries
scripts/figures/     figure scripts
experiments/         one script per experiment of the paper
tests/               unit tests
```

## Citation

```bibtex
@inproceedings{su2026umlr,
  title     = {Umbilic Multinomial Logistic Regression},
  author    = {Su, Zihan and Sebe, Nicu and Sch{\"o}lkopf, Bernhard and Chen, Ziheng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This code is released under the [MIT License](LICENSE).
