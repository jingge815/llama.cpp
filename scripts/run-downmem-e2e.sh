#!/usr/bin/env bash
set -euo pipefail

LLAMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNMEM_ROOT="${DOWNMEM_ROOT:-/home/fjg/src/downmem}"
DOWNMEM_BUILD="${DOWNMEM_BUILD:-${DOWNMEM_ROOT}/build-rv}"
LLAMA_BUILD="${LLAMA_BUILD:-${LLAMA_ROOT}/build-downmem}"
DPU_BIN="${GGML_DOWNMEM_DPU_BIN:-${DOWNMEM_BUILD}/devApp/rvbins/LLAMA_GEMV_F32}"
MODEL="${MODEL:-${LLAMA_ROOT}/models/stories15M-q4_0.gguf}"
N_PREDICT="${N_PREDICT:-1}"
PROMPT="${PROMPT:-Once upon a time}"
LOG_DIR="${LOG_DIR:-${LLAMA_BUILD}/downmem-e2e-logs}"

mkdir -p "${LOG_DIR}"

CPU_OUT="${LOG_DIR}/cpu.out"
CPU_ERR="${LOG_DIR}/cpu.err"
DOWNMEM_OUT="${LOG_DIR}/downmem.out"
DOWNMEM_ERR="${LOG_DIR}/downmem.err"
GEMV_LOG="${LOG_DIR}/downmem-gemv.log"
GGML_LOG="${LOG_DIR}/ggml-test.log"
DIFF_LOG="${LOG_DIR}/cpu-vs-downmem.diff"

step() {
  printf '\n==> %s\n' "$*"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_file "${MODEL}" "model"
require_file "${DOWNMEM_BUILD}/libdmmShared.so" "downmem shared library"

step "Build downmem LLAMA_GEMV_F32 kernel, host test, and shared runtime"
cmake --build "${DOWNMEM_BUILD}" --target dmmShared rvLLAMA_GEMV_F32 dmmLLAMA_GEMV_F32 -j"$(nproc)"

if [[ ! -x "${DPU_BIN}" ]]; then
  echo "missing DPU binary: ${DPU_BIN}" >&2
  exit 1
fi

step "Run standalone downmem GEMV test"
DMM_NR_SIM_THRDS="${DMM_NR_SIM_THRDS:-4}" \
  "${DOWNMEM_BUILD}/dmmLLAMA_GEMV_F32" "${DPU_BIN}" 2>&1 | tee "${GEMV_LOG}"
grep -E "LLAMA_GEMV_F32 passed" "${GEMV_LOG}" >/dev/null

step "Configure llama.cpp with GGML_DOWNMEM=ON"
cmake -S "${LLAMA_ROOT}" -B "${LLAMA_BUILD}" \
  -DGGML_DOWNMEM=ON \
  -DGGML_DOWNMEM_ROOT="${DOWNMEM_ROOT}" \
  -DGGML_DOWNMEM_LIB="${GGML_DOWNMEM_LIB:-${DOWNMEM_BUILD}/libdmmShared.so}" \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

step "Build llama.cpp downmem test and completion tool"
cmake --build "${LLAMA_BUILD}" --target test-downmem-backend llama-completion -j"$(nproc)"

step "Run ggml-level downmem correctness test"
DMM_NR_SIM_THRDS="${DMM_NR_SIM_THRDS:-4}" \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN="${DPU_BIN}" \
GGML_DOWNMEM_NR_DPUS="${GGML_DOWNMEM_NR_DPUS:-4}" \
GGML_DOWNMEM_VALIDATE=1 \
GGML_DOWNMEM_VERBOSE=1 \
"${LLAMA_BUILD}/bin/test-downmem-backend" 2>&1 | tee "${GGML_LOG}"
grep -E "test-downmem-backend passed" "${GGML_LOG}" >/dev/null
grep -E "executed op=1" "${GGML_LOG}" >/dev/null

COMMON_ARGS=(
  -m "${MODEL}"
  -p "${PROMPT}"
  -n "${N_PREDICT}"
  -t 1 -tb 1
  -s 42
  --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1
  --no-display-prompt
  --no-warmup
  --no-context-shift
  -fit off
  --log-verbosity 1
)

step "Run CPU baseline completion"
"${LLAMA_BUILD}/bin/llama-completion" "${COMMON_ARGS[@]}" >"${CPU_OUT}" 2>"${CPU_ERR}"

step "Run downmem-enabled completion"
DMM_NR_SIM_THRDS="${DMM_NR_SIM_THRDS:-4}" \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN="${DPU_BIN}" \
GGML_DOWNMEM_NR_DPUS="${GGML_DOWNMEM_NR_DPUS:-4}" \
GGML_DOWNMEM_MAX_OPS="${GGML_DOWNMEM_MAX_OPS:-1}" \
GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1 \
GGML_DOWNMEM_VERBOSE=1 \
"${LLAMA_BUILD}/bin/llama-completion" \
  "${COMMON_ARGS[@]}" \
  --device Downmem >"${DOWNMEM_OUT}" 2>"${DOWNMEM_ERR}"

step "Verify downmem offload log"
grep -E "claiming MUL_MAT|executed op=1" "${DOWNMEM_ERR}" >/dev/null
grep -E "executed op=1" "${DOWNMEM_ERR}"

step "Compare CPU and downmem generated stdout"
if ! diff -u "${CPU_OUT}" "${DOWNMEM_OUT}" >"${DIFF_LOG}"; then
  echo "CPU and downmem stdout differ. See ${DIFF_LOG}" >&2
  exit 1
fi

step "E2E passed"
echo "logs: ${LOG_DIR}"
echo "generated stdout:"
cat "${DOWNMEM_OUT}"
