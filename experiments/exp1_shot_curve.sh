#!/usr/bin/env bash
# Experiment 1, parameter-matched H/H^2/S/S^2 study across shots (Table 5).
#
# All four heads have K(d+2) trainable parameters; frozen CL2N features;
# 1/2/4/8/16/32 labeled images per class; paired seeds across heads.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-cifar100 flowers102 tiny_imagenet}"
BACKBONES="${BACKBONES:-vit_b_16 convnext_tiny}"
SHOTS="${SHOTS:-1 2 4 8 16 32}"

for dataset in ${DATASETS}; do
  for backbone in ${BACKBONES}; do
    for shots in ${SHOTS}; do
      for seed in ${SEEDS}; do
        train shot_curve "${dataset}" "${backbone}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${MATCHED_H[@]}"
        train shot_curve "${dataset}" "${backbone}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${MATCHED_H2[@]}"
        train shot_curve "${dataset}" "${backbone}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${MATCHED_S[@]}"
        train shot_curve "${dataset}" "${backbone}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${MATCHED_S2[@]}"
      done
    done
  done
done

run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/attribution_stats.py" --study shot_curve \
  --input-root "${OUTPUT_ROOT}/shot_curve" --output-dir "${OUTPUT_ROOT}/reports/shot_curve"
