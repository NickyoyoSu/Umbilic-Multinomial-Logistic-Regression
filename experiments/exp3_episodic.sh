#!/usr/bin/env bash
# Experiment 3 (Table 7): validation-selected UMLR endpoint on full-way episodes.
#
# 1. Run the four adapted endpoints (UMLR-H, UMLR^2-H, UMLR-S, UMLR^2-S) on
#    validation classes for every dataset, shot, and validation seed.
# 2. Lock one endpoint with scripts/select_episodic_endpoint.py.
# 3. Evaluate only ProtoNet, SimpleShot-CL2N, and the locked endpoint on the
#    test classes with new seeds.
# Optionally (FOUR_ENDPOINT_TEST=1) also run the descriptive four-endpoint test
# comparison reported in the appendix.
#
# miniImageNet, CIFAR-FS, and FC100 are loaded from the Hugging Face hub; CUB
# and tieredImageNet archives are downloaded from pancake/few_shot_datasets.
# Frozen ViT-B/16 features are cached under ${OUTPUT_ROOT}/feature_cache on the
# first run; caching tieredImageNet needs a large-memory host.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PROTOCOL="${PROTOCOL:-${REPO_ROOT}/experiments/episodic_protocol.json}"
read_protocol() { "${PYTHON}" -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))[sys.argv[2]])))" "${PROTOCOL}" "$1"; }
DATASETS="${DATASETS:-$(read_protocol datasets)}"
SHOTS="${SHOTS:-$(read_protocol shots)}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-$(read_protocol validation_seeds)}"
TEST_SEEDS="${TEST_SEEDS:-$(read_protocol test_seeds)}"
FOUR_ENDPOINT_TEST="${FOUR_ENDPOINT_TEST:-0}"
FEATURE_ROOT="${FEATURE_ROOT:-${OUTPUT_ROOT}/feature_cache}"

EPISODIC=("${PYTHON}" "${REPO_ROOT}/scripts/run_episodic.py" --data-root "${DATA_ROOT}" --backbone vit_b_16 --pretrained
  --way full --query-per-class 15 --episodes 200 --feature-mode cl2n --adapt-steps 100 --adapt-lr 0.01
  --download-pancake-archives --num-workers "${NUM_WORKERS}")

episode() {  # episode <split> <dataset> <shot> <seed> <output-root> <methods...>
  local split="$1" dataset="$2" shot="$3" seed="$4" root="$5"
  shift 5
  run "${EPISODIC[@]}" --dataset "${dataset}" --split "${split}" --shot "${shot}" --seed "${seed}" \
    --feature-cache "${FEATURE_ROOT}/${dataset}_vit_b_16_${split}.pt" \
    --output-root "${root}" --run-name "${dataset}_${split}_full_${shot}shot_s${seed}" --methods "$@"
}

# 1. Validation runs.
for dataset in ${DATASETS}; do
  for shot in ${SHOTS}; do
    for seed in ${VALIDATION_SEEDS}; do
      episode validation "${dataset}" "${shot}" "${seed}" "${OUTPUT_ROOT}/episodic_validation" \
        adapt_h adapt_h2 adapt_s adapt_s2
    done
  done
done

# 2. Lock the endpoint using validation results only.
MANIFEST="${OUTPUT_ROOT}/reports/episodic_selection_manifest.json"
run "${PYTHON}" "${REPO_ROOT}/scripts/select_episodic_endpoint.py" \
  --input-root "${OUTPUT_ROOT}/episodic_validation" --protocol "${PROTOCOL}" --manifest "${MANIFEST}" \
  --report-md "${OUTPUT_ROOT}/reports/episodic_selection.md"
if [[ "${DRY_RUN}" == "1" ]]; then
  SELECTED="adapt_s"
else
  SELECTED="$("${PYTHON}" -c "import json,sys; m=json.load(open(sys.argv[1])); assert m['status']=='locked'; print(m['selected_method'])" "${MANIFEST}")"
fi
echo "Locked endpoint: ${SELECTED}"

# 3. Locked test runs with new seeds.
for dataset in ${DATASETS}; do
  for shot in ${SHOTS}; do
    for seed in ${TEST_SEEDS}; do
      episode test "${dataset}" "${shot}" "${seed}" "${OUTPUT_ROOT}/episodic_test" \
        protonet simpleshot_cl2n "${SELECTED}"
    done
  done
done
run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_episodic.py" --split test \
  --input-root "${OUTPUT_ROOT}/episodic_test" --report-md "${OUTPUT_ROOT}/reports/episodic_test.md"

# Optional descriptive comparison of all four endpoints on the test classes.
if [[ "${FOUR_ENDPOINT_TEST}" == "1" ]]; then
  for dataset in ${DATASETS}; do
    for shot in ${SHOTS}; do
      for seed in ${FOUR_ENDPOINT_SEEDS:-0 1 2 3 4}; do
        episode test "${dataset}" "${shot}" "${seed}" "${OUTPUT_ROOT}/episodic_four_endpoint" \
          protonet simpleshot_cl2n adapt_h adapt_h2 adapt_s adapt_s2
      done
    done
  done
  run "${PYTHON}" "${REPO_ROOT}/scripts/analysis/summarize_episodic.py" --split test \
    --input-root "${OUTPUT_ROOT}/episodic_four_endpoint" --report-md "${OUTPUT_ROOT}/reports/episodic_four_endpoint.md"
fi
