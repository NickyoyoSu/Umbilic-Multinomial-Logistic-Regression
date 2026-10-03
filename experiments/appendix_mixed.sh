#!/usr/bin/env bash
# Appendix: exploratory classwise mixed heads (Mixed-H and Mixed-H^2).
#
# Hard straight-through gate between a hyperplane logit and UMLR^2-S, gate
# initialized at hyperplane probability GATE_INIT_RATIO (default 0.5). The
# endpoint columns of the appendix table use the same mixed-head recipe.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-cifar100 flowers102 tiny_imagenet}"
BACKBONES="${BACKBONES:-vit_b_16 convnext_tiny}"
SHOTS="${SHOTS:-4 16}"
RUN_FULL_DATA="${RUN_FULL_DATA:-1}"
HEAD_H2=(--head-type hyperplane --hyperplane-score-mode signed_square)

for dataset in ${DATASETS}; do
  for backbone in ${BACKBONES}; do
    for seed in ${SEEDS}; do
      regimes=()
      for shots in ${SHOTS}; do regimes+=("${shots}"); done
      if [[ "${RUN_FULL_DATA}" == "1" ]]; then regimes+=(0); fi
      for shots in "${regimes[@]}"; do
        if [[ "${shots}" == "0" ]]; then recipe=("${FULL_FINETUNE[@]}"); else recipe=("${MIXED_FROZEN[@]}"); fi
        for head in H H2 S2 MIX_H MIX_H2; do
          case "${head}" in
            H) head_args=("${HEAD_H[@]}") ;;
            H2) head_args=("${HEAD_H2[@]}") ;;
            S2) head_args=("${HEAD_S2[@]}") ;;
            MIX_H) head_args=("${HEAD_MIX_H[@]}") ;;
            MIX_H2) head_args=("${HEAD_MIX_H2[@]}") ;;
          esac
          train mixed "${dataset}" "${backbone}" "${seed}" "${shots}" "${recipe[@]}" "${head_args[@]}"
        done
      done
    done
  done
done

run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_runs.py" \
  --input-root "${OUTPUT_ROOT}/mixed" --report-md "${OUTPUT_ROOT}/reports/mixed.md"
run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/mixed_gate_audit.py" \
  --input-root "${OUTPUT_ROOT}/mixed" --report-md "${OUTPUT_ROOT}/reports/mixed_gates.md"
