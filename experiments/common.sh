#!/usr/bin/env bash
# Shared settings for the experiment scripts in this directory.
#
# Every variable can be overridden from the environment, for example
#   SEEDS="0" DATA_ROOT=/path/to/data bash experiments/exp1_head_swap.sh
# Set DRY_RUN=1 to print the commands without running them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN=("${PYTHON}" "${REPO_ROOT}/scripts/train.py")

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

# --- Training recipes -------------------------------------------------------

# Shared input pipeline: ImageNet-pretrained torchvision weights, 224x224
# inputs, RandomResizedCrop + flip + RandAugment(2, 9), and CL2N features
# (training-mean centering followed by L2 normalization).
COMMON=(
  --pretrained
  --batch-size 64
  --optimizer adamw
  --scheduler cosine
  --min-lr 1e-6
  --label-smoothing 0.1
  --train-augmentation vit_repro
  --randaugment-num-ops 2
  --randaugment-magnitude 9
  --input-size 224
  --num-workers "${NUM_WORKERS}"
  --feature-normalization
  --feature-centering train_mean
)

# Head-swap suite (Tables 2 and 4, low-shot rows): frozen backbone, 60 epochs.
HEADSWAP_FROZEN=("${COMMON[@]}" --epochs 60 --lr 1e-3 --head-lr-factor 1.0 --weight-decay 1e-4 --warmup-epochs 3 --freeze-backbone)

# Full-data suite (Table 2, full-data rows): end-to-end fine-tuning, 100 epochs.
FULL_FINETUNE=("${COMMON[@]}" --epochs 100 --lr 5e-5 --head-lr-factor 10.0 --weight-decay 0.05 --warmup-epochs 5)

# Mixed-head suite (appendix), low-shot rows: frozen backbone, 60 epochs.
MIXED_FROZEN=("${COMMON[@]}" --epochs 60 --lr 1e-3 --head-lr-factor 1.0 --weight-decay 0.05 --warmup-epochs 5 --freeze-backbone)

# Parameter-matched recipe (Table 3, radius controls, sensitivity, frozen-versus-
# fine-tuned, full-data and long-tail controls): every head has an explicit
# classwise positive scale initialized at 1.
MATCHED_BASE=("${COMMON[@]}" --epochs 60 --weight-decay 1e-4 --warmup-epochs 3 --radius-lr-factor 1.0 --alpha-init-effective 1.0)
MATCHED_FROZEN=("${MATCHED_BASE[@]}" --lr 1e-3 --head-lr-factor 1.0 --freeze-backbone)
MATCHED_FINETUNE=("${MATCHED_BASE[@]}" --lr 1e-5 --head-lr-factor 100)

# --- Heads -----------------------------------------------------------------

# Head-swap and full-data suites.
HEAD_H=(--head-type hyperplane)
HEAD_PROTO=(--head-type prototype)
HEAD_S=(--head-type sphere --sphere-parameterization ppt_decoupled_inside --sphere-boundary-mode distance
  --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.9)
HEAD_S2=(--head-type sphere --sphere-parameterization ppt_decoupled_inside --sphere-boundary-mode signed_square
  --umlr-alpha-mode identity
  --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.9)

# Parameter-matched heads: K(d+2) trainable parameters each.
MATCHED_H=(--head-type hyperplane --hyperplane-score-mode linear --hyperplane-alpha-scope classwise)
MATCHED_H2=(--head-type hyperplane --hyperplane-score-mode signed_square --hyperplane-alpha-scope classwise)
MATCHED_S=(--head-type sphere --sphere-parameterization ppt_decoupled_inside --sphere-boundary-mode distance
  --alpha-scope classwise --radius-scope classwise
  --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.8)
MATCHED_S2=(--head-type sphere --sphere-parameterization ppt_decoupled_inside --sphere-boundary-mode signed_square
  --alpha-scope classwise --radius-scope classwise
  --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.8)

# Exploratory mixed heads: classwise hard gate, UMLR^2-S spherical side.
MIXED_SPHERE_SIDE=(--sphere-parameterization ppt_decoupled_inside --sphere-boundary-mode signed_square
  --umlr-alpha-mode identity
  --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.9
  --gate-mode classwise --gate-type hard --gate-init-ratio "${GATE_INIT_RATIO:-0.5}")
HEAD_MIX_H=(--head-type mixed --hyperplane-score-mode linear "${MIXED_SPHERE_SIDE[@]}")
HEAD_MIX_H2=(--head-type mixed --hyperplane-score-mode signed_square "${MIXED_SPHERE_SIDE[@]}")

# train <tag> <dataset> <backbone> <seed> <shots> <recipe/head args...>
train() {
  local tag="$1" dataset="$2" backbone="$3" seed="$4" shots="$5"
  shift 5
  run "${TRAIN[@]}" --dataset "${dataset}" --backbone "${backbone}" --data-root "${DATA_ROOT}" \
    --seed "${seed}" --train-shots-per-class "${shots}" \
    --run-tag "${tag}" --output-root "${OUTPUT_ROOT}/${tag}" "$@"
}
