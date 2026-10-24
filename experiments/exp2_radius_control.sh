#!/usr/bin/env bash
# Experiment 2, radius-only control for UMLR^2-S on ViT-B/16.
#
# Same class-mean centers, link, scale, split, and optimizer; only the radius
# changes: r = 0 (prototype limit), one shared radius, a fixed q80 radius, and a
# learned q80 radius. The Table 6 comparison comes from exp1_head_swap.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-cifar100 flowers102 tiny_imagenet}"
BACKBONE="${BACKBONE:-vit_b_16}"
SHOTS="${SHOTS:-4 16}"

S2_BASE=(--head-type sphere --sphere-parameterization ppt_decoupled_inside --sphere-boundary-mode signed_square
  --alpha-scope classwise)
Q80=(--feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.8)

for dataset in ${DATASETS}; do
  for shots in ${SHOTS}; do
    for seed in ${SEEDS}; do
      train radius_control "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${S2_BASE[@]}" \
        --radius-scope fixed --fixed-radius 0 "${Q80[@]}" --feature-stat-centers-only
      train radius_control "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${S2_BASE[@]}" \
        --radius-scope shared "${Q80[@]}"
      train radius_control "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${S2_BASE[@]}" \
        --radius-scope classwise "${Q80[@]}" --freeze-radius
      train radius_control "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${S2_BASE[@]}" \
        --radius-scope classwise "${Q80[@]}"
    done
  done
done

run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/attribution_stats.py" --study radius --seed 20260724 \
  --input-root "${OUTPUT_ROOT}/radius_control" --output-dir "${OUTPUT_ROOT}/reports/radius_control"
