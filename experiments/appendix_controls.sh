#!/usr/bin/env bash
# Appendix controls on the parameter-matched H/H^2/S/S^2 family (ViT-B/16 unless noted).
#
#   PARTS=sensitivity  radius-initialization, radius-LR, and scale-initialization
#                      sensitivity of UMLR^2-S at 4/16 shots (reference: q80,
#                      radius LR x1, scale 1, i.e. the UMLR^2-S runs of exp1_shot_curve.sh)
#   PARTS=finetune     frozen versus end-to-end fine-tuned backbones on five datasets
#   PARTS=longtail     balanced full data (ViT-B/16 and ConvNeXt-Tiny) and CIFAR-100-LT
#   PARTS=large_k      compute benchmark for K up to 10,000 (CUDA)
# Run a subset with, e.g., PARTS="sensitivity longtail".
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PARTS="${PARTS:-sensitivity finetune longtail large_k}"
BACKBONE="${BACKBONE:-vit_b_16}"

S2_ARGS=(--head-type sphere --sphere-parameterization ppt_decoupled_inside --sphere-boundary-mode signed_square
  --alpha-scope classwise --radius-scope classwise)

has_part() { [[ " ${PARTS} " == *" $1 "* ]]; }

if has_part sensitivity; then
  for dataset in cifar100 flowers102 tiny_imagenet; do
    for shots in 4 16; do
      for seed in ${SEEDS}; do
        for quantile in 0.5 0.9; do
          train sensitivity "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${S2_ARGS[@]}" \
            --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile "${quantile}"
        done
        for radius_lr in 0.1 10; do
          train sensitivity "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${S2_ARGS[@]}" \
            --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.8 \
            --radius-lr-factor "${radius_lr}"
        done
        for alpha_init in 0.5 2; do
          train sensitivity "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${MATCHED_FROZEN[@]}" "${S2_ARGS[@]}" \
            --feature-stat-init centroid_radius --feature-stat-radius-mode quantile --feature-stat-radius-quantile 0.8 \
            --alpha-init-effective "${alpha_init}"
        done
      done
    done
  done
  run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_runs.py" \
    --input-root "${OUTPUT_ROOT}/sensitivity" "${OUTPUT_ROOT}/shot_curve" --report-md "${OUTPUT_ROOT}/reports/sensitivity.md"
fi

if has_part finetune; then
  for dataset in cifar100 flowers102 tiny_imagenet oxford_iiit_pet food101; do
    for shots in 4 16; do
      for seed in ${SEEDS}; do
        for protocol in frozen finetuned; do
          if [[ "${protocol}" == "frozen" ]]; then recipe=("${MATCHED_FROZEN[@]}"); else recipe=("${MATCHED_FINETUNE[@]}"); fi
          for head in H H2 S S2; do
            case "${head}" in
              H) head_args=("${MATCHED_H[@]}") ;;
              H2) head_args=("${MATCHED_H2[@]}") ;;
              S) head_args=("${MATCHED_S[@]}") ;;
              S2) head_args=("${MATCHED_S2[@]}") ;;
            esac
            train frozen_vs_finetuned "${dataset}" "${BACKBONE}" "${seed}" "${shots}" "${recipe[@]}" "${head_args[@]}" \
              --write-classwise-diagnostics
          done
        done
      done
    done
  done
  run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_runs.py" \
    --input-root "${OUTPUT_ROOT}/frozen_vs_finetuned" --report-md "${OUTPUT_ROOT}/reports/frozen_vs_finetuned.md"
fi

if has_part longtail; then
  for seed in ${SEEDS}; do
    for head in H H2 S S2; do
      case "${head}" in
        H) head_args=("${MATCHED_H[@]}") ;;
        H2) head_args=("${MATCHED_H2[@]}") ;;
        S) head_args=("${MATCHED_S[@]}") ;;
        S2) head_args=("${MATCHED_S2[@]}") ;;
      esac
      for backbone in vit_b_16 convnext_tiny; do
        train full_balanced cifar100 "${backbone}" "${seed}" 0 "${MATCHED_FROZEN[@]}" "${head_args[@]}" \
          --write-classwise-diagnostics
      done
      for ratio in 0.1 0.01; do
        train longtail cifar100 vit_b_16 "${seed}" 0 "${MATCHED_FROZEN[@]}" "${head_args[@]}" \
          --train-imbalance-type exp --train-imbalance-ratio "${ratio}" --write-classwise-diagnostics
      done
    done
  done
  run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_runs.py" \
    --input-root "${OUTPUT_ROOT}/full_balanced" "${OUTPUT_ROOT}/longtail" --report-md "${OUTPUT_ROOT}/reports/full_longtail.md"
fi

if has_part large_k; then
  run "${PYTHON}" "${REPO_ROOT}/scripts/benchmark_large_k.py" --classes 1000 5000 10000 \
    --feature-dim 768 --batch-size 64 --output "${OUTPUT_ROOT}/reports/large_k.json"
fi
