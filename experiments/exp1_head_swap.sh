#!/usr/bin/env bash
# Experiment 1 (Table 2) and Experiment 2 (Table 4).
#
# Low-shot rows: frozen CL2N features, 4/16 labeled images per class,
#   UMLR-H, UMLR-S, and UMLR^2-S (Table 2), plus the zero-radius limit of
#   UMLR^2-S (Table 4: UMLR^2-S with every radius fixed at zero, paired with the
#   UMLR^2-S runs by seed).
# Full-data rows: end-to-end fine-tuning of UMLR-H, UMLR-S, and UMLR^2-S.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATASETS="${DATASETS:-cifar100 flowers102 tiny_imagenet}"
BACKBONES="${BACKBONES:-vit_b_16 convnext_tiny}"
SHOTS="${SHOTS:-4 16}"
RUN_FULL_DATA="${RUN_FULL_DATA:-1}"

for dataset in ${DATASETS}; do
  for backbone in ${BACKBONES}; do
    for seed in ${SEEDS}; do
      for shots in ${SHOTS}; do
        train exp1_lowshot "${dataset}" "${backbone}" "${seed}" "${shots}" "${HEADSWAP_FROZEN[@]}" "${HEAD_H[@]}"
        train exp1_lowshot "${dataset}" "${backbone}" "${seed}" "${shots}" "${HEADSWAP_FROZEN[@]}" "${HEAD_S[@]}"
        train exp1_lowshot "${dataset}" "${backbone}" "${seed}" "${shots}" "${HEADSWAP_FROZEN[@]}" "${HEAD_S2[@]}"
        train exp1_lowshot "${dataset}" "${backbone}" "${seed}" "${shots}" "${HEADSWAP_FROZEN[@]}" "${HEAD_S2_R0[@]}"
      done
      if [[ "${RUN_FULL_DATA}" == "1" ]]; then
        train exp1_full "${dataset}" "${backbone}" "${seed}" 0 "${FULL_FINETUNE[@]}" "${HEAD_H[@]}"
        train exp1_full "${dataset}" "${backbone}" "${seed}" 0 "${FULL_FINETUNE[@]}" "${HEAD_S[@]}"
        train exp1_full "${dataset}" "${backbone}" "${seed}" 0 "${FULL_FINETUNE[@]}" "${HEAD_S2[@]}"
      fi
    done
  done
done

run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_runs.py" \
  --input-root "${OUTPUT_ROOT}/exp1_lowshot" "${OUTPUT_ROOT}/exp1_full" \
  --report-md "${OUTPUT_ROOT}/reports/exp1_head_swap.md"
