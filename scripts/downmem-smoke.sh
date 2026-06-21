#!/usr/bin/env bash
set -euo pipefail

LLAMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNMEM_ROOT="${DOWNMEM_ROOT:-/home/fjg/src/downmem}"
DOWNMEM_BUILD="${DOWNMEM_BUILD:-${DOWNMEM_ROOT}/build-rv}"
LLAMA_BUILD="${LLAMA_BUILD:-${LLAMA_ROOT}/build-downmem}"
DPU_BIN="${GGML_DOWNMEM_DPU_BIN:-${DOWNMEM_BUILD}/devApp/rvbins/LLAMA_GEMV_F32}"
MODEL="${MODEL:-${LLAMA_ROOT}/models/stories15M-q4_0.gguf}"

if [[ ! -f "${MODEL}" ]]; then
  echo "missing model: ${MODEL}" >&2
  exit 1
fi

cmake --build "${DOWNMEM_BUILD}" --target rvLLAMA_GEMV_F32 dmmLLAMA_GEMV_F32 -j"$(nproc)"

if [[ ! -x "${DPU_BIN}" ]]; then
  echo "missing DPU binary: ${DPU_BIN}" >&2
  exit 1
fi

cmake -S "${LLAMA_ROOT}" -B "${LLAMA_BUILD}" \
  -DGGML_DOWNMEM=ON \
  -DGGML_DOWNMEM_ROOT="${DOWNMEM_ROOT}" \
  -DGGML_DOWNMEM_LIB="${GGML_DOWNMEM_LIB:-${DOWNMEM_BUILD}/libdmmShared.so}" \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${LLAMA_BUILD}" --target test-downmem-backend llama-completion -j"$(nproc)"

TEST_LOG="$(mktemp)"
LLAMA_LOG="$(mktemp)"
trap 'rm -f "${TEST_LOG}" "${LLAMA_LOG}"' EXIT

DMM_NR_SIM_THRDS="${DMM_NR_SIM_THRDS:-4}" \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN="${DPU_BIN}" \
GGML_DOWNMEM_NR_DPUS="${GGML_DOWNMEM_NR_DPUS:-4}" \
GGML_DOWNMEM_MAX_COLS="${GGML_DOWNMEM_MAX_COLS:-128}" \
GGML_DOWNMEM_ENABLE_SCALE="${GGML_DOWNMEM_ENABLE_SCALE:-1}" \
GGML_DOWNMEM_VALIDATE=1 \
GGML_DOWNMEM_VERBOSE=1 \
"${LLAMA_BUILD}/bin/test-downmem-backend" 2>&1 | tee "${TEST_LOG}"

grep -E "test-downmem-backend passed" "${TEST_LOG}" >/dev/null
grep -E "executed op=1" "${TEST_LOG}" >/dev/null
grep -E "executed op=.*m=" "${TEST_LOG}" >/dev/null
grep -E "kind=SCALE" "${TEST_LOG}" >/dev/null

DMM_NR_SIM_THRDS="${DMM_NR_SIM_THRDS:-4}" \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN="${DPU_BIN}" \
GGML_DOWNMEM_NR_DPUS="${GGML_DOWNMEM_NR_DPUS:-4}" \
GGML_DOWNMEM_MAX_OPS="${GGML_DOWNMEM_MAX_OPS:-1}" \
GGML_DOWNMEM_MAX_SCALE_OPS="${GGML_DOWNMEM_MAX_SCALE_OPS:-2}" \
GGML_DOWNMEM_MAX_COLS="${GGML_DOWNMEM_MAX_COLS:-128}" \
GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1 \
GGML_DOWNMEM_ENABLE_SCALE="${GGML_DOWNMEM_ENABLE_SCALE:-1}" \
GGML_DOWNMEM_VERBOSE=1 \
"${LLAMA_BUILD}/bin/llama-completion" \
  -m "${MODEL}" \
  -p "Once upon a time" \
  -n "${N_PREDICT:-1}" \
  -t 1 -tb 1 \
  -s 42 \
  --temp 0.8 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
  --backend-sampling \
  --no-display-prompt \
  --no-warmup \
  --no-context-shift \
  --device Downmem \
  -fit off \
  --log-verbosity 3 2>&1 | tee "${LLAMA_LOG}"

grep -E "claiming MUL_MAT|executed op=1" "${LLAMA_LOG}" >/dev/null
grep -E "executed op=1.*m=" "${LLAMA_LOG}" >/dev/null
grep -E "kind=SCALE" "${LLAMA_LOG}" >/dev/null
