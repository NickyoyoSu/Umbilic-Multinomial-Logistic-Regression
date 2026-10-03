#!/usr/bin/env bash
# Experiment 4 (Table 6): ImageNet-1K head retraining with angular-margin controls.
#
# 1. Cache frozen ImageNet-1K-pretrained ViT-B/16 penultimate features of all
#    training and validation images (multi-GPU: set NPROC_PER_NODE > 1).
# 2. Train UMLR-H, UMLR^2-H, UMLR-S, UMLR^2-S, CosFace, and ArcFace heads from
#    scratch for 20 epochs; report final-epoch validation accuracy.
#
# IMAGENET_ROOT must contain train/ and val/ in ImageFolder layout (val images
# sorted into class folders).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

IMAGENET_ROOT="${IMAGENET_ROOT:-${DATA_ROOT}/imagenet}"
CACHE_DIR="${CACHE_DIR:-${OUTPUT_ROOT}/feature_cache/imagenet_vit_b_16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ ! -f "${CACHE_DIR}/feature_stats.pt" ]]; then
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}")
  else
    LAUNCH=("${PYTHON}")
  fi
  run "${LAUNCH[@]}" "${REPO_ROOT}/scripts/extract_imagenet_features.py" --data-root "${IMAGENET_ROOT}" \
    --cache-dir "${CACHE_DIR}" --backbone vit_b_16 --batch-size 128 --num-workers "${NUM_WORKERS}" --amp
fi

for method in h h2 s s2 cosface arcface; do
  for seed in ${SEEDS}; do
    run "${PYTHON}" "${REPO_ROOT}/scripts/train_imagenet_head.py" --cache-dir "${CACHE_DIR}" \
      --output-dir "${OUTPUT_ROOT}/imagenet_heads/${method}_seed${seed}" --method "${method}" --seed "${seed}" \
      --epochs 20 --batch-size 4096 --eval-batch-size 8192 --lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.1 \
      --angular-scale 30 --angular-margin 0.35 --num-workers 2
  done
done

run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_imagenet.py" \
  --input-root "${OUTPUT_ROOT}/imagenet_heads" --report-md "${OUTPUT_ROOT}/reports/imagenet.md"
