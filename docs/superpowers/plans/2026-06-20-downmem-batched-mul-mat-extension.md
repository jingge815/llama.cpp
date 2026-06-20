# Downmem Batched MUL_MAT Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current Downmem offload path from single-column `GGML_OP_MUL_MAT` GEMV to multi-column F32 `GGML_OP_MUL_MAT` while keeping full llama.cpp inference correct against CPU.

**Architecture:** Keep one Downmem DPU binary and one ggml backend runtime path. Reuse the existing host staging model, extend the MRAM layout from one right-hand-side vector to `m` right-hand-side vectors, and gather a `[n, m]` F32 result back into ggml's tensor layout. Do not add RMS_NORM, ADD, MUL, ROPE, GLU, or FLASH_ATTN_EXT execution in this pass because the current scheduler offload hook naturally selects ops with host weight buffers, and the current backend runtime loads a single DPU program.

**Tech Stack:** CMake, C/C++17, ggml backend API, llama.cpp, Downmem host API (`dpu_alloc`, `dpu_load`, `dpu_push_xfer`, `dpu_broadcast_to`, `dpu_launch`), RV DPU runtime (`mram_read`, `mram_write`).

---

## Scope Decision

The current backend supports only this shape:

```text
src0: [k, n, 1, 1]
src1: [k, 1, 1, 1]
dst:  [n, 1, 1, 1]
```

This plan extends the same op to:

```text
src0: [k, n, 1, 1]
src1: [k, m, 1, 1]
dst:  [n, m, 1, 1]
```

This is the safest next operator expansion because it stays inside `GGML_OP_MUL_MAT`, uses the existing llama.cpp offload scheduling path, and matches real prompt/batched graph shapes observed with `export-graph-ops`.

RMS_NORM is a good next candidate after this plan, but it should be handled in a separate design because it is not selected through the same "host weight buffer" offload path and would need a clear multi-kernel or unified-kernel strategy. ADD and MUL are useful only as simple F32 demos unless their scheduling and tensor movement policy is specified. GLU, ROPE, and FLASH_ATTN_EXT stay out of scope.

## File Map

- Modify: `/home/fjg/src/downmem/devApp/LLAMA_GEMV_F32.c`
  - Add `m` to `DPU_INPUT_ARGUMENTS`.
  - Compute all `m` output columns for each assigned row.
  - Keep target name `rvLLAMA_GEMV_F32` and binary path `devApp/rvbins/LLAMA_GEMV_F32`.
- Modify: `/home/fjg/src/downmem/hostApp/LLAMA_GEMV_F32.c`
  - Extend standalone correctness cases from GEMV to batched MUL_MAT.
  - Keep target name `dmmLLAMA_GEMV_F32`.
- Modify: `/home/fjg/src/llama.cpp/ggml/src/ggml-downmem/ggml-downmem.cpp`
  - Rename shape helpers from GEMV-specific names to MUL_MAT F32 names.
  - Accept `src1->ne[1] >= 1`.
  - Stage `src1` as `m` contiguous K-length vectors.
  - Transfer and gather a `[max_rows, m]` output block per DPU.
  - Replace process-global offload claim state with device-context state.
- Modify: `/home/fjg/src/llama.cpp/tests/test-downmem-backend.cpp`
  - Add multi-column correctness cases.
- Modify: `/home/fjg/src/llama.cpp/scripts/run-downmem-e2e.sh`
  - Keep the same Downmem targets.
  - Add log checks for `m=`.
- Modify: `/home/fjg/src/llama.cpp/scripts/downmem-smoke.sh`
  - Add the same runtime variable defaults used by the E2E runner.
- Modify: `/home/fjg/src/llama.cpp/docs/backend/DOWNMEM.md`
  - Document the new supported shape and remaining unsupported ops.
- Modify: `/home/fjg/src/llama.cpp/README-downmem.md`
  - Update the summary and validation notes.

Commit commands are intentionally omitted because `AGENTS.md` requires explicit human approval before any commit. Use `git diff --check` and `git status --short` checkpoints instead.

### Task 1: Baseline Verification

**Files:**
- Read: `/home/fjg/src/llama.cpp/ggml/src/ggml-downmem/ggml-downmem.cpp`
- Read: `/home/fjg/src/llama.cpp/tests/test-downmem-backend.cpp`
- Read: `/home/fjg/src/downmem/devApp/LLAMA_GEMV_F32.c`
- Read: `/home/fjg/src/downmem/hostApp/LLAMA_GEMV_F32.c`

- [ ] **Step 1: Confirm clean llama.cpp state**

Run:

```bash
cd /home/fjg/src/llama.cpp
git status --short
```

Expected:

```text
?? docs/superpowers/
```

If additional tracked files appear, inspect them before editing. Do not revert user changes.

- [ ] **Step 2: Confirm existing downmem state**

Run:

```bash
cd /home/fjg/src/downmem
git status --short
```

Expected: the pre-existing `.gitignore` modification and patch files may appear. Do not remove them.

- [ ] **Step 3: Build the current standalone Downmem target**

Run:

```bash
cd /home/fjg/src/downmem
cmake --build build-rv --target dmmShared rvLLAMA_GEMV_F32 dmmLLAMA_GEMV_F32 -j"$(nproc)"
```

Expected:

```text
Built target dmmShared
Built target rvLLAMA_GEMV_F32
Built target dmmLLAMA_GEMV_F32
```

- [ ] **Step 4: Run the current standalone GEMV test**

Run:

```bash
cd /home/fjg/src/downmem
DMM_NR_SIM_THRDS=4 ./build-rv/dmmLLAMA_GEMV_F32 ./build-rv/devApp/rvbins/LLAMA_GEMV_F32
```

Expected:

```text
LLAMA_GEMV_F32 passed
```

- [ ] **Step 5: Build the current llama.cpp Downmem test**

Run:

```bash
cd /home/fjg/src/llama.cpp
cmake -S . -B build-downmem \
  -DGGML_DOWNMEM=ON \
  -DGGML_DOWNMEM_ROOT=/home/fjg/src/downmem \
  -DGGML_DOWNMEM_LIB=/home/fjg/src/downmem/build-rv/libdmmShared.so \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-downmem --target test-downmem-backend llama-completion -j"$(nproc)"
```

Expected:

```text
Built target test-downmem-backend
Built target llama-completion
```

- [ ] **Step 6: Run the current ggml-level backend test**

Run:

```bash
cd /home/fjg/src/llama.cpp
DMM_NR_SIM_THRDS=4 \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN=/home/fjg/src/downmem/build-rv/devApp/rvbins/LLAMA_GEMV_F32 \
GGML_DOWNMEM_NR_DPUS=4 \
GGML_DOWNMEM_VALIDATE=1 \
GGML_DOWNMEM_VERBOSE=1 \
./build-downmem/bin/test-downmem-backend
```

Expected:

```text
test-downmem-backend passed
```

### Task 2: Add Failing Multi-Column Downmem Host Test

**Files:**
- Modify: `/home/fjg/src/downmem/hostApp/LLAMA_GEMV_F32.c`

- [ ] **Step 1: Replace the host test with multi-column cases**

Apply this replacement to `/home/fjg/src/downmem/hostApp/LLAMA_GEMV_F32.c`:

```c
#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint32_t k;
  uint32_t k_pad;
  uint32_t n_rows;
  uint32_t max_rows;
  uint32_t m;
  uint32_t a_offset;
  uint32_t x_offset;
  uint32_t y_offset;
} llama_gemv_f32_args_t;

typedef struct {
  uint32_t rows;
  uint32_t row_offset;
} dpu_rows_t;

static uint32_t align_up_u32(uint32_t v, uint32_t a) {
  return ((v + a - 1) / a) * a;
}

static float pattern_a(uint32_t row, uint32_t col) {
  const int v = (int)((row * 17u + col * 13u + 7u) % 23u) - 11;
  return (float)v * 0.03125f;
}

static float pattern_x(uint32_t out_col, uint32_t k_col) {
  const int v = (int)((out_col * 11u + k_col * 19u + 5u) % 29u) - 14;
  return (float)v * 0.0625f;
}

static void host_mul_mat(const float *a, const float *x, float *y,
                         uint32_t n, uint32_t k, uint32_t m) {
  for (uint32_t out_col = 0; out_col < m; ++out_col) {
    for (uint32_t row = 0; row < n; ++row) {
      float sum = 0.0f;
      for (uint32_t col = 0; col < k; ++col) {
        sum += a[row * k + col] * x[out_col * k + col];
      }
      y[(size_t)out_col * n + row] = sum;
    }
  }
}

static int nearly_equal(float got, float expected) {
  const float diff = fabsf(got - expected);
  const float scale = fmaxf(1.0f, fabsf(expected));
  return diff <= 1.0e-4f || diff <= scale * 1.0e-4f;
}

static int run_case(uint32_t n, uint32_t k, uint32_t m, uint32_t nr_dpus,
                    const char *dpu_binary) {
  struct dpu_set_t set;

  DMM_VERIFY(dpu_alloc(nr_dpus, NULL, &set));
  DMM_VERIFY(dpu_load(set, dpu_binary, NULL));

  uint32_t actual_dpus = 0;
  DMM_VERIFY(dpu_get_nr_dpus(set, &actual_dpus));
  assert(actual_dpus == nr_dpus);

  const uint32_t k_pad = align_up_u32(k, 2);
  uint32_t max_rows = 0;
  dpu_rows_t *layout = calloc(nr_dpus, sizeof(*layout));
  llama_gemv_f32_args_t *args = calloc(nr_dpus, sizeof(*args));
  assert(layout != NULL && args != NULL);

  for (uint32_t i = 0; i < nr_dpus; ++i) {
    const uint32_t base = n / nr_dpus;
    const uint32_t rest = n % nr_dpus;
    layout[i].rows = base + (i < rest ? 1u : 0u);
    layout[i].row_offset = i * base + (i < rest ? i : rest);
    if (layout[i].rows > max_rows) {
      max_rows = layout[i].rows;
    }
  }
  if (max_rows == 0) {
    max_rows = 1;
  }

  const uint32_t a_floats_per_dpu = max_rows * k_pad;
  const uint32_t x_floats_per_dpu = k_pad * m;
  const uint32_t y_floats_per_dpu = max_rows * m;
  const uint32_t a_bytes = a_floats_per_dpu * sizeof(float);
  const uint32_t x_bytes = x_floats_per_dpu * sizeof(float);
  const uint32_t y_bytes = y_floats_per_dpu * sizeof(float);

  float *a = calloc((size_t)n * k, sizeof(float));
  float *x = calloc((size_t)m * k, sizeof(float));
  float *x_padded = calloc(x_floats_per_dpu, sizeof(float));
  float *y_ref = calloc((size_t)n * m, sizeof(float));
  float *y_dpu_padded = calloc((size_t)nr_dpus * y_floats_per_dpu, sizeof(float));
  float *a_padded = calloc((size_t)nr_dpus * a_floats_per_dpu, sizeof(float));
  assert(a != NULL && x != NULL && x_padded != NULL && y_ref != NULL &&
         y_dpu_padded != NULL && a_padded != NULL);

  for (uint32_t row = 0; row < n; ++row) {
    for (uint32_t col = 0; col < k; ++col) {
      a[row * k + col] = pattern_a(row, col);
    }
  }
  for (uint32_t out_col = 0; out_col < m; ++out_col) {
    for (uint32_t col = 0; col < k; ++col) {
      const float value = pattern_x(out_col, col);
      x[out_col * k + col] = value;
      x_padded[out_col * k_pad + col] = value;
    }
  }
  host_mul_mat(a, x, y_ref, n, k, m);

  for (uint32_t d = 0; d < nr_dpus; ++d) {
    for (uint32_t row = 0; row < layout[d].rows; ++row) {
      const uint32_t global_row = layout[d].row_offset + row;
      memcpy(a_padded + (size_t)d * a_floats_per_dpu + row * k_pad,
             a + (size_t)global_row * k, k * sizeof(float));
    }
    args[d] = (llama_gemv_f32_args_t){
        .k = k,
        .k_pad = k_pad,
        .n_rows = layout[d].rows,
        .max_rows = max_rows,
        .m = m,
        .a_offset = 0,
        .x_offset = a_bytes,
        .y_offset = a_bytes + x_bytes,
    };
  }

  struct dpu_set_t each;
  uint32_t idx = 0;
  DPU_FOREACH(set, each, idx) { DMM_VERIFY(dpu_prepare_xfer(each, args + idx)); }
  DMM_VERIFY(dpu_push_xfer(set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                           sizeof(llama_gemv_f32_args_t), DPU_XFER_DEFAULT));

  idx = 0;
  DPU_FOREACH(set, each, idx) {
    DMM_VERIFY(dpu_prepare_xfer(each,
                                a_padded + (size_t)idx * a_floats_per_dpu));
  }
  DMM_VERIFY(dpu_push_xfer(set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                           a_bytes, DPU_XFER_DEFAULT));

  DMM_VERIFY(dpu_broadcast_to(set, DPU_MRAM_HEAP_POINTER_NAME, a_bytes,
                              x_padded, x_bytes, DPU_XFER_DEFAULT));

  DMM_VERIFY(dpu_launch(set, DPU_SYNCHRONOUS));

  idx = 0;
  DPU_FOREACH(set, each, idx) {
    DMM_VERIFY(dpu_prepare_xfer(each,
                                y_dpu_padded + (size_t)idx * y_floats_per_dpu));
  }
  DMM_VERIFY(dpu_push_xfer(set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                           a_bytes + x_bytes, y_bytes, DPU_XFER_DEFAULT));

  int ok = 1;
  for (uint32_t d = 0; d < nr_dpus; ++d) {
    for (uint32_t row = 0; row < layout[d].rows; ++row) {
      const uint32_t global_row = layout[d].row_offset + row;
      for (uint32_t out_col = 0; out_col < m; ++out_col) {
        const float got = y_dpu_padded[(size_t)d * y_floats_per_dpu +
                                       (size_t)row * m + out_col];
        const float expected = y_ref[(size_t)out_col * n + global_row];
        if (!nearly_equal(got, expected)) {
          fprintf(stderr,
                  "case n=%u k=%u m=%u dpus=%u row=%u out_col=%u expected=%g got=%g\n",
                  n, k, m, nr_dpus, global_row, out_col, expected, got);
          ok = 0;
          goto done;
        }
      }
    }
  }

done:
  free(a);
  free(x);
  free(x_padded);
  free(y_ref);
  free(y_dpu_padded);
  free(a_padded);
  free(layout);
  free(args);
  DMM_VERIFY(dpu_free(set));
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/LLAMA_GEMV_F32\n", argv[0]);
    return 2;
  }

  const char *dpu_binary = argv[1];
  int failed = 0;
  failed |= run_case(8, 16, 1, 1, dpu_binary);
  failed |= run_case(17, 31, 1, 4, dpu_binary);
  failed |= run_case(17, 31, 3, 4, dpu_binary);
  failed |= run_case(33, 64, 5, 6, dpu_binary);
  failed |= run_case(3, 7, 4, 8, dpu_binary);

  if (failed != 0) {
    fprintf(stderr, "LLAMA_GEMV_F32 failed\n");
    return 1;
  }

  printf("LLAMA_GEMV_F32 passed\n");
  return 0;
}
```

- [ ] **Step 2: Build and verify the expected failure**

Run:

```bash
cd /home/fjg/src/downmem
cmake --build build-rv --target dmmLLAMA_GEMV_F32 -j"$(nproc)"
DMM_NR_SIM_THRDS=4 ./build-rv/dmmLLAMA_GEMV_F32 ./build-rv/devApp/rvbins/LLAMA_GEMV_F32
```

Expected: the command exits non-zero and prints a mismatch for a case with `m` greater than `1`.

### Task 3: Extend the Downmem DPU Kernel

**Files:**
- Modify: `/home/fjg/src/downmem/devApp/LLAMA_GEMV_F32.c`

- [ ] **Step 1: Replace the DPU kernel with multi-column computation**

Apply this replacement to `/home/fjg/src/downmem/devApp/LLAMA_GEMV_F32.c`:

```c
#include "moredefs.h"
#include <alloc.h>
#include <stdint.h>

typedef struct {
  uint32_t k;
  uint32_t k_pad;
  uint32_t n_rows;
  uint32_t max_rows;
  uint32_t m;
  uint32_t a_offset;
  uint32_t x_offset;
  uint32_t y_offset;
} llama_gemv_f32_args_t;

__host llama_gemv_f32_args_t DPU_INPUT_ARGUMENTS;

ALL_THREADS_BARRIER_INIT();

#define LLAMA_GEMV_CHUNK_FLOATS 64u

static uint32_t rows_start_for_tasklet(uint32_t tasklet_id, uint32_t n_rows) {
  const uint32_t base = n_rows / NR_TASKLETS;
  const uint32_t rest = n_rows % NR_TASKLETS;
  return tasklet_id * base + (tasklet_id < rest ? tasklet_id : rest);
}

static uint32_t rows_count_for_tasklet(uint32_t tasklet_id, uint32_t n_rows) {
  const uint32_t base = n_rows / NR_TASKLETS;
  const uint32_t rest = n_rows % NR_TASKLETS;
  return base + (tasklet_id < rest ? 1u : 0u);
}

int main(void) {
  const uint32_t tasklet_id = me();
  const uint32_t k = DPU_INPUT_ARGUMENTS.k;
  const uint32_t k_pad = DPU_INPUT_ARGUMENTS.k_pad;
  const uint32_t n_rows = DPU_INPUT_ARGUMENTS.n_rows;
  const uint32_t m = DPU_INPUT_ARGUMENTS.m;
  const uint32_t a_offset = DPU_INPUT_ARGUMENTS.a_offset;
  const uint32_t x_offset = DPU_INPUT_ARGUMENTS.x_offset;
  const uint32_t y_offset = DPU_INPUT_ARGUMENTS.y_offset;

  float a_buf[LLAMA_GEMV_CHUNK_FLOATS];
  float x_buf[LLAMA_GEMV_CHUNK_FLOATS];
  float y_buf[1];

  all_threads_barrier_wait();

  const uint32_t row_start = rows_start_for_tasklet(tasklet_id, n_rows);
  const uint32_t row_count = rows_count_for_tasklet(tasklet_id, n_rows);
  const uint32_t row_end = row_start + row_count;

  for (uint32_t row = row_start; row < row_end; ++row) {
    for (uint32_t out_col = 0; out_col < m; ++out_col) {
      float sum = 0.0f;

      for (uint32_t col = 0; col < k; col += LLAMA_GEMV_CHUNK_FLOATS) {
        uint32_t chunk = k - col;
        if (chunk > LLAMA_GEMV_CHUNK_FLOATS) {
          chunk = LLAMA_GEMV_CHUNK_FLOATS;
        }

        const uint32_t bytes = chunk * sizeof(float);
        const uintptr_t a_addr = (uintptr_t)DPU_MRAM_HEAP_POINTER + a_offset +
                                 (row * k_pad + col) * sizeof(float);
        const uintptr_t x_addr = (uintptr_t)DPU_MRAM_HEAP_POINTER + x_offset +
                                 (out_col * k_pad + col) * sizeof(float);

        mram_read((__mram_ptr void const *)a_addr, a_buf, bytes);
        mram_read((__mram_ptr void const *)x_addr, x_buf, bytes);

        for (uint32_t i = 0; i < chunk; ++i) {
          sum += a_buf[i] * x_buf[i];
        }
      }

      y_buf[0] = sum;
      const uintptr_t y_addr = (uintptr_t)DPU_MRAM_HEAP_POINTER + y_offset +
                               (row * m + out_col) * sizeof(float);
      mram_write(y_buf, (__mram_ptr void *)y_addr, sizeof(float));
    }
  }

  return 0;
}
```

- [ ] **Step 2: Build and run the standalone test**

Run:

```bash
cd /home/fjg/src/downmem
cmake --build build-rv --target rvLLAMA_GEMV_F32 dmmLLAMA_GEMV_F32 -j"$(nproc)"
DMM_NR_SIM_THRDS=4 ./build-rv/dmmLLAMA_GEMV_F32 ./build-rv/devApp/rvbins/LLAMA_GEMV_F32
```

Expected:

```text
LLAMA_GEMV_F32 passed
```

- [ ] **Step 3: Check formatting and diff**

Run:

```bash
cd /home/fjg/src/downmem
git diff -- devApp/LLAMA_GEMV_F32.c hostApp/LLAMA_GEMV_F32.c
git diff --check -- devApp/LLAMA_GEMV_F32.c hostApp/LLAMA_GEMV_F32.c
```

Expected: `git diff --check` exits `0`.

### Task 4: Add Failing Multi-Column llama.cpp Backend Test

**Files:**
- Modify: `/home/fjg/src/llama.cpp/tests/test-downmem-backend.cpp`

- [ ] **Step 1: Replace the ggml-level test**

Apply this replacement to `/home/fjg/src/llama.cpp/tests/test-downmem-backend.cpp`:

```cpp
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float pattern_a(int row, int col) {
    const int v = (row * 17 + col * 13 + 7) % 23 - 11;
    return (float) v * 0.03125f;
}

static float pattern_x(int out_col, int col) {
    const int v = (out_col * 11 + col * 19 + 5) % 29 - 14;
    return (float) v * 0.0625f;
}

static void fill_tensor_2d_f32(ggml_tensor * t, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        float * dst = (float *) ((char *) t->data + row * t->nb[1]);
        for (int col = 0; col < cols; ++col) {
            dst[col] = pattern_a(row, col);
        }
    }
}

static void fill_rhs_f32(ggml_tensor * t, int k, int m) {
    for (int out_col = 0; out_col < m; ++out_col) {
        float * dst = (float *) ((char *) t->data + out_col * t->nb[1]);
        for (int col = 0; col < k; ++col) {
            dst[col] = pattern_x(out_col, col);
        }
    }
}

static std::vector<float> run_graph(ggml_backend_t backend, int n, int k, int m) {
    const size_t mem_size = 16u * 1024u * 1024u;
    ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    fill_tensor_2d_f32(a, n, k);
    fill_rhs_f32(x, k, m);

    ggml_tensor * y = ggml_mul_mat(ctx, a, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed: %d\n", (int) status);
        std::exit(1);
    }

    std::vector<float> result((size_t) n * m);
    for (int out_col = 0; out_col < m; ++out_col) {
        for (int row = 0; row < n; ++row) {
            result[(size_t) out_col * n + row] =
                *(float *) ((char *) y->data + row * y->nb[0] + out_col * y->nb[1]);
        }
    }

    ggml_free(ctx);
    return result;
}

static bool compare_case(const std::vector<float> & got, const std::vector<float> & expected,
                         int n, int m) {
    for (int out_col = 0; out_col < m; ++out_col) {
        for (int row = 0; row < n; ++row) {
            const int i = out_col * n + row;
            const float diff = std::fabs(got[i] - expected[i]);
            const float scale = std::fmax(1.0f, std::fabs(expected[i]));
            if (!(diff <= 1.0e-4f || diff <= scale * 1.0e-4f)) {
                fprintf(stderr, "row=%d out_col=%d expected=%g got=%g diff=%g\n",
                        row, out_col, (double) expected[i], (double) got[i], (double) diff);
                return false;
            }
        }
    }
    return true;
}

int main(void) {
    const char * enabled = std::getenv("GGML_DOWNMEM");
    const char * bin = std::getenv("GGML_DOWNMEM_DPU_BIN");
    if (enabled == nullptr || std::strcmp(enabled, "0") == 0 || bin == nullptr || bin[0] == '\0') {
        std::puts("SKIP test-downmem-backend: GGML_DOWNMEM and GGML_DOWNMEM_DPU_BIN are required");
        return 77;
    }

    struct test_case {
        int n;
        int k;
        int m;
    };

    const test_case cases[] = {
        { 17, 31, 1 },
        { 17, 31, 3 },
        { 33, 64, 5 },
    };

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        fprintf(stderr, "failed to init CPU backend\n");
        return 1;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Downmem");
    if (dev == nullptr) {
        fprintf(stderr, "Downmem backend device not registered\n");
        ggml_backend_free(cpu);
        return 1;
    }

    ggml_backend_t downmem = ggml_backend_dev_init(dev, nullptr);
    if (downmem == nullptr) {
        fprintf(stderr, "failed to init Downmem backend\n");
        ggml_backend_free(cpu);
        return 1;
    }

    for (const test_case & tc : cases) {
        std::vector<float> expected = run_graph(cpu, tc.n, tc.k, tc.m);
        std::vector<float> got = run_graph(downmem, tc.n, tc.k, tc.m);
        if (!compare_case(got, expected, tc.n, tc.m)) {
            ggml_backend_free(downmem);
            ggml_backend_free(cpu);
            return 1;
        }
    }

    ggml_backend_free(downmem);
    ggml_backend_free(cpu);

    std::puts("test-downmem-backend passed");
    return 0;
}
```

- [ ] **Step 2: Build and verify the expected failure**

Run:

```bash
cd /home/fjg/src/llama.cpp
cmake --build build-downmem --target test-downmem-backend -j"$(nproc)"
DMM_NR_SIM_THRDS=4 \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN=/home/fjg/src/downmem/build-rv/devApp/rvbins/LLAMA_GEMV_F32 \
GGML_DOWNMEM_NR_DPUS=4 \
GGML_DOWNMEM_VALIDATE=1 \
GGML_DOWNMEM_VERBOSE=1 \
./build-downmem/bin/test-downmem-backend
```

Expected: the command exits non-zero before the llama backend implementation is updated, because the backend still rejects `m > 1`.

### Task 5: Extend the llama.cpp Downmem Backend

**Files:**
- Modify: `/home/fjg/src/llama.cpp/ggml/src/ggml-downmem/ggml-downmem.cpp`

- [ ] **Step 1: Update the DPU argument struct and add device claim context**

In `/home/fjg/src/llama.cpp/ggml/src/ggml-downmem/ggml-downmem.cpp`, replace the existing `llama_gemv_f32_args_t` and add the device context directly after `ggml_backend_downmem_context`:

```cpp
struct llama_gemv_f32_args_t {
    uint32_t k;
    uint32_t k_pad;
    uint32_t n_rows;
    uint32_t max_rows;
    uint32_t m;
    uint32_t a_offset;
    uint32_t x_offset;
    uint32_t y_offset;
};

struct dpu_rows_t {
    uint32_t rows;
    uint32_t row_offset;
};

struct ggml_backend_downmem_context {
    bool loaded = false;
    bool failed = false;
    uint32_t nr_dpus = 0;
    std::string binary_path;
    dpu_set_t set = {};
    int executed_ops = 0;
};

struct ggml_backend_downmem_device_context {
    int claimed_ops = 0;
};
```

- [ ] **Step 2: Replace the src1 shape helper**

Replace `ggml_downmem_src1_is_plain_f32_vector` with:

```cpp
static bool ggml_downmem_src1_is_plain_f32_matrix(const ggml_tensor * t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           t->nb[0] == (int64_t) sizeof(float) &&
           t->nb[1] >= t->nb[0] * t->ne[0] &&
           t->ne[2] == 1 &&
           t->ne[3] == 1;
}
```

- [ ] **Step 3: Replace the supported-op check**

Replace `ggml_downmem_supports_mul_mat_gemv` with:

```cpp
static bool ggml_downmem_supports_mul_mat_f32(const ggml_tensor * op, bool require_runtime_ready) {
    if (!ggml_downmem_env_enabled()) {
        return false;
    }
    if (require_runtime_ready && !ggml_downmem_file_exists(ggml_downmem_dpu_bin())) {
        return false;
    }
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!ggml_downmem_src0_can_stage_to_f32(src0) || !ggml_downmem_src1_is_plain_f32_matrix(src1)) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    if (op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1]) {
        return false;
    }
    if (op->nb[0] != (int64_t) sizeof(float) || op->nb[1] < op->nb[0] * op->ne[0]) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }

    const int max_cols = ggml_downmem_env_i32("GGML_DOWNMEM_MAX_COLS", 128);
    if (max_cols > 0 && src1->ne[1] > max_cols) {
        return false;
    }

    return true;
}
```

- [ ] **Step 4: Replace the src1 staging helper**

Replace `ggml_downmem_stage_src1_vector` with:

```cpp
static void ggml_downmem_stage_src1_matrix(const ggml_tensor * src1, float * x, uint32_t k, uint32_t m, uint32_t k_pad) {
    for (uint32_t out_col = 0; out_col < m; ++out_col) {
        const char * src_col = (const char *) src1->data + out_col * src1->nb[1];
        float * dst_col = x + (size_t) out_col * k_pad;
        if (src1->nb[0] == (int64_t) sizeof(float)) {
            std::memcpy(dst_col, src_col, k * sizeof(float));
            continue;
        }

        for (uint32_t col = 0; col < k; ++col) {
            const char * src = src_col + col * src1->nb[0];
            std::memcpy(dst_col + col, src, sizeof(float));
        }
    }
}
```

- [ ] **Step 5: Replace the validation helper**

Replace `ggml_downmem_validate_result` with:

```cpp
static bool ggml_downmem_validate_result(const float * x, const float * a_rows, const float * y,
                                         uint32_t k, uint32_t n, uint32_t m, uint32_t k_pad) {
    const char * enabled = std::getenv("GGML_DOWNMEM_VALIDATE");
    if (enabled == nullptr || std::strcmp(enabled, "0") == 0) {
        return true;
    }

    for (uint32_t out_col = 0; out_col < m; ++out_col) {
        const float * x_col = x + (size_t) out_col * k_pad;
        for (uint32_t row = 0; row < n; ++row) {
            float expected = 0.0f;
            const float * a_row = a_rows + (size_t) row * k_pad;
            for (uint32_t col = 0; col < k; ++col) {
                expected += a_row[col] * x_col[col];
            }

            const float got = y[(size_t) out_col * n + row];
            const float diff = std::fabs(got - expected);
            const float scale = std::max(1.0f, std::fabs(expected));
            if (!(diff <= 1.0e-4f || diff <= scale * 1.0e-4f)) {
                GGML_LOG_ERROR("%s: validation failed row=%u out_col=%u expected=%g got=%g diff=%g\n",
                               __func__, row, out_col, (double) expected, (double) got, (double) diff);
                return false;
            }
        }
    }

    return true;
}
```

- [ ] **Step 6: Rename and extend the execution function**

Rename `ggml_downmem_mul_mat_gemv` to `ggml_downmem_mul_mat_f32`. Inside that function, make these exact changes:

```cpp
if (!ggml_downmem_supports_mul_mat_f32(dst, true)) {
    GGML_LOG_ERROR("%s: scheduled unsupported op\n", __func__);
    return false;
}
```

Use these dimensions:

```cpp
const uint32_t k = (uint32_t) src0->ne[0];
const uint32_t n = (uint32_t) src0->ne[1];
const uint32_t m = (uint32_t) src1->ne[1];
const uint32_t k_pad = ggml_downmem_align_up_u32(k, 2);
```

Use these per-DPU buffer sizes:

```cpp
const uint32_t a_floats_per_dpu = max_rows * k_pad;
const uint32_t x_floats_per_dpu = k_pad * m;
const uint32_t y_floats_per_dpu = max_rows * m;
const uint32_t a_bytes = a_floats_per_dpu * sizeof(float);
const uint32_t x_bytes = x_floats_per_dpu * sizeof(float);
const uint32_t y_bytes = y_floats_per_dpu * sizeof(float);
```

Use these host vectors:

```cpp
std::vector<float> a_padded((size_t) ctx->nr_dpus * a_floats_per_dpu, 0.0f);
std::vector<float> x(x_floats_per_dpu, 0.0f);
std::vector<float> y_padded((size_t) ctx->nr_dpus * y_floats_per_dpu, 0.0f);
std::vector<float> y((size_t) n * m, 0.0f);
std::vector<llama_gemv_f32_args_t> args(ctx->nr_dpus);

ggml_downmem_stage_src1_matrix(src1, x.data(), k, m, k_pad);
```

When filling DPU args, use:

```cpp
args[d] = llama_gemv_f32_args_t {
    /* .k        = */ k,
    /* .k_pad    = */ k_pad,
    /* .n_rows   = */ layout[d].rows,
    /* .max_rows = */ max_rows,
    /* .m        = */ m,
    /* .a_offset = */ 0,
    /* .x_offset = */ a_bytes,
    /* .y_offset = */ a_bytes + x_bytes,
};
```

When reading DPU output, use:

```cpp
idx = 0;
DPU_FOREACH(ctx->set, each, idx) {
    if (dpu_prepare_xfer(each, y_padded.data() + (size_t) idx * y_floats_per_dpu) != DPU_OK) {
        GGML_LOG_ERROR("%s: dpu_prepare_xfer y failed\n", __func__);
        return false;
    }
}
if (dpu_push_xfer(ctx->set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, a_bytes + x_bytes,
                  y_bytes, DPU_XFER_DEFAULT) != DPU_OK) {
    GGML_LOG_ERROR("%s: y transfer failed\n", __func__);
    return false;
}

for (uint32_t d = 0; d < ctx->nr_dpus; ++d) {
    for (uint32_t row = 0; row < layout[d].rows; ++row) {
        const uint32_t global_row = layout[d].row_offset + row;
        for (uint32_t out_col = 0; out_col < m; ++out_col) {
            y[(size_t) out_col * n + global_row] =
                y_padded[(size_t) d * y_floats_per_dpu + (size_t) row * m + out_col];
        }
    }
}
```

Validate and write back with:

```cpp
if (!ggml_downmem_validate_result(x.data(), a_rows.data(), y.data(), k, n, m, k_pad)) {
    return false;
}

for (uint32_t out_col = 0; out_col < m; ++out_col) {
    for (uint32_t row = 0; row < n; ++row) {
        float * dst_elem = (float *) ((char *) dst->data + row * dst->nb[0] + out_col * dst->nb[1]);
        *dst_elem = y[(size_t) out_col * n + row];
    }
}

++ctx->executed_ops;
ggml_downmem_verbose_log("%s: executed op=%d k=%u n=%u m=%u dpus=%u max_rows=%u bytes=%" PRIu64 "\n",
                         __func__, ctx->executed_ops, k, n, m, ctx->nr_dpus, max_rows, total_mram_bytes);
```

- [ ] **Step 7: Update graph dispatch and support hooks**

Replace the graph dispatch call:

```cpp
case GGML_OP_MUL_MAT:
    if (!ggml_downmem_mul_mat_f32(ctx, node)) {
        return GGML_STATUS_FAILED;
    }
    break;
```

Replace device support:

```cpp
static bool ggml_backend_downmem_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_downmem_supports_mul_mat_f32(op, false);
}
```

Replace the offload hook:

```cpp
static bool ggml_backend_downmem_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto * ctx = (ggml_backend_downmem_device_context *) dev->context;
    if (ctx == nullptr) {
        return false;
    }

    const int max_ops = ggml_downmem_env_i32("GGML_DOWNMEM_MAX_OPS", 1 << 30);
    const bool supported = ggml_downmem_supports_mul_mat_f32(op, true);

    if (!supported || ctx->claimed_ops >= max_ops) {
        return false;
    }

    ++ctx->claimed_ops;
    ggml_downmem_verbose_log("%s: claiming MUL_MAT k=%" PRId64 " n=%" PRId64 " m=%" PRId64 " claimed=%d max=%d\n",
                             __func__, op->src[0]->ne[0], op->src[0]->ne[1], op->src[1]->ne[1],
                             ctx->claimed_ops, max_ops);
    return true;
}
```

- [ ] **Step 8: Wire the device context into the registry**

Replace `ggml_backend_downmem_reg_context` and `ggml_backend_downmem_reg` with:

```cpp
struct ggml_backend_downmem_reg_context {
    ggml_backend_device device;
    ggml_backend_downmem_device_context device_ctx;
};

static const char * ggml_backend_downmem_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Downmem";
}

static size_t ggml_backend_downmem_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_downmem_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    auto * ctx = (ggml_backend_downmem_reg_context *) reg->context;
    return &ctx->device;
}

static const ggml_backend_reg_i ggml_backend_downmem_reg_i = {
    /* .get_name         = */ ggml_backend_downmem_reg_get_name,
    /* .get_device_count = */ ggml_backend_downmem_reg_get_device_count,
    /* .get_device       = */ ggml_backend_downmem_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_downmem_reg(void) {
    static ggml_backend_downmem_reg_context ctx = {};
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_downmem_reg_i,
        /* .context     = */ &ctx,
    };

    ctx.device.iface = ggml_backend_downmem_device_i;
    ctx.device.reg = &reg;
    ctx.device.context = &ctx.device_ctx;
    return &reg;
}
```

- [ ] **Step 9: Build and run the backend test**

Run:

```bash
cd /home/fjg/src/llama.cpp
cmake --build build-downmem --target test-downmem-backend -j"$(nproc)"
DMM_NR_SIM_THRDS=4 \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN=/home/fjg/src/downmem/build-rv/devApp/rvbins/LLAMA_GEMV_F32 \
GGML_DOWNMEM_NR_DPUS=4 \
GGML_DOWNMEM_VALIDATE=1 \
GGML_DOWNMEM_VERBOSE=1 \
./build-downmem/bin/test-downmem-backend
```

Expected:

```text
test-downmem-backend passed
```

stderr should contain at least one `executed op=` line with `m=3` or `m=5`.

### Task 6: Update Runner Scripts

**Files:**
- Modify: `/home/fjg/src/llama.cpp/scripts/run-downmem-e2e.sh`
- Modify: `/home/fjg/src/llama.cpp/scripts/downmem-smoke.sh`

- [ ] **Step 1: Update `run-downmem-e2e.sh` runtime environment**

In `/home/fjg/src/llama.cpp/scripts/run-downmem-e2e.sh`, add `GGML_DOWNMEM_MAX_COLS` to both test and completion invocations:

```bash
GGML_DOWNMEM_MAX_COLS="${GGML_DOWNMEM_MAX_COLS:-128}" \
```

For the ggml-level test block, the environment should be:

```bash
DMM_NR_SIM_THRDS="${DMM_NR_SIM_THRDS:-4}" \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN="${DPU_BIN}" \
GGML_DOWNMEM_NR_DPUS="${GGML_DOWNMEM_NR_DPUS:-4}" \
GGML_DOWNMEM_MAX_COLS="${GGML_DOWNMEM_MAX_COLS:-128}" \
GGML_DOWNMEM_VALIDATE=1 \
GGML_DOWNMEM_VERBOSE=1 \
"${LLAMA_BUILD}/bin/test-downmem-backend" 2>&1 | tee "${GGML_LOG}"
```

For the downmem completion block, the environment should be:

```bash
DMM_NR_SIM_THRDS="${DMM_NR_SIM_THRDS:-4}" \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN="${DPU_BIN}" \
GGML_DOWNMEM_NR_DPUS="${GGML_DOWNMEM_NR_DPUS:-4}" \
GGML_DOWNMEM_MAX_OPS="${GGML_DOWNMEM_MAX_OPS:-1}" \
GGML_DOWNMEM_MAX_COLS="${GGML_DOWNMEM_MAX_COLS:-128}" \
GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1 \
GGML_DOWNMEM_VERBOSE=1 \
"${LLAMA_BUILD}/bin/llama-completion" \
  "${COMMON_ARGS[@]}" \
  --device Downmem >"${DOWNMEM_OUT}" 2>"${DOWNMEM_ERR}"
```

Update the log checks to accept the new log form:

```bash
grep -E "claiming MUL_MAT|executed op=1" "${DOWNMEM_ERR}" >/dev/null
grep -E "executed op=1.*m=" "${DOWNMEM_ERR}"
```

- [ ] **Step 2: Update `downmem-smoke.sh` runtime environment**

In `/home/fjg/src/llama.cpp/scripts/downmem-smoke.sh`, add `GGML_DOWNMEM_MAX_COLS="${GGML_DOWNMEM_MAX_COLS:-128}" \` to the `test-downmem-backend` invocation and the `llama-completion --device Downmem` invocation.

- [ ] **Step 3: Run the E2E runner**

Run:

```bash
cd /home/fjg/src/llama.cpp
DOWNMEM_ROOT=/home/fjg/src/downmem \
MODEL=/home/fjg/src/llama.cpp/models/stories15M-q4_0.gguf \
N_PREDICT=1 \
PROMPT="Once upon a time" \
scripts/run-downmem-e2e.sh
```

Expected:

```text
==> E2E passed
```

Expected logs:

```text
build-downmem/downmem-e2e-logs/downmem.err
build-downmem/downmem-e2e-logs/ggml-test.log
```

At least one Downmem log line should show `m=`. The CPU and Downmem stdout diff must be empty.

### Task 7: Update Documentation

**Files:**
- Modify: `/home/fjg/src/llama.cpp/docs/backend/DOWNMEM.md`
- Modify: `/home/fjg/src/llama.cpp/README-downmem.md`

- [ ] **Step 1: Update supported operator documentation**

In `/home/fjg/src/llama.cpp/docs/backend/DOWNMEM.md`, replace the "supported op" shape text with:

```markdown
The backend supports a narrow F32 `GGML_OP_MUL_MAT` shape:

- `src0` is a 2D matrix with shape `[k, n, 1, 1]`.
- `src1` is an F32 matrix with shape `[k, m, 1, 1]`.
- `dst` is F32 with shape `[n, m, 1, 1]`.
- `src0->ne[0] == src1->ne[0]`.
- `dst->ne[0] == src0->ne[1]`.
- `dst->ne[1] == src1->ne[1]`.
- `src0` is either F32 or host-dequantizable through `ggml_get_type_traits(type)->to_float`.
- Quantized `src0` still requires `GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1`.
- `GGML_DOWNMEM_MAX_COLS` limits `m` and defaults to `128`.

The DPU binary is still named `LLAMA_GEMV_F32` for compatibility with the existing scripts, but the kernel now handles multiple RHS vectors in one launch.
```

- [ ] **Step 2: Add unsupported operator rationale**

Add this paragraph to the limitations section:

```markdown
RMS_NORM, ADD, MUL, GLU, ROPE, and FLASH_ATTN_EXT are not executed by this backend in this extension. RMS_NORM is the strongest next candidate, but it needs a separate scheduler and kernel-management design because the current offload hook is naturally reached for ops that consume host weight buffers, while RMS_NORM usually consumes activation tensors. ADD and MUL are feasible as contiguous F32 elementwise demos, but they would not materially improve the current LLM path without a clear policy for when the scheduler should place activation-only ops on Downmem.
```

- [ ] **Step 3: Update README summary**

In `/home/fjg/src/llama.cpp/README-downmem.md`, update the summary bullet from "GEMV only" to:

```markdown
- The experimental Downmem backend offloads a narrow F32 `GGML_OP_MUL_MAT` path. It supports both single-vector GEMV and small batched RHS cases where `src1` has shape `[k, m]`.
```

- [ ] **Step 4: Check docs diff**

Run:

```bash
cd /home/fjg/src/llama.cpp
git diff -- docs/backend/DOWNMEM.md README-downmem.md scripts/run-downmem-e2e.sh scripts/downmem-smoke.sh
git diff --check -- docs/backend/DOWNMEM.md README-downmem.md scripts/run-downmem-e2e.sh scripts/downmem-smoke.sh
```

Expected: `git diff --check` exits `0`.

### Task 8: Final Verification

**Files:**
- Verify: `/home/fjg/src/downmem/devApp/LLAMA_GEMV_F32.c`
- Verify: `/home/fjg/src/downmem/hostApp/LLAMA_GEMV_F32.c`
- Verify: `/home/fjg/src/llama.cpp/ggml/src/ggml-downmem/ggml-downmem.cpp`
- Verify: `/home/fjg/src/llama.cpp/tests/test-downmem-backend.cpp`
- Verify: `/home/fjg/src/llama.cpp/scripts/run-downmem-e2e.sh`
- Verify: `/home/fjg/src/llama.cpp/docs/backend/DOWNMEM.md`

- [ ] **Step 1: Run standalone Downmem correctness**

Run:

```bash
cd /home/fjg/src/downmem
cmake --build build-rv --target dmmShared rvLLAMA_GEMV_F32 dmmLLAMA_GEMV_F32 -j"$(nproc)"
DMM_NR_SIM_THRDS=4 ./build-rv/dmmLLAMA_GEMV_F32 ./build-rv/devApp/rvbins/LLAMA_GEMV_F32
```

Expected:

```text
LLAMA_GEMV_F32 passed
```

- [ ] **Step 2: Run llama.cpp backend correctness**

Run:

```bash
cd /home/fjg/src/llama.cpp
cmake --build build-downmem --target test-downmem-backend llama-completion -j"$(nproc)"
DMM_NR_SIM_THRDS=4 \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN=/home/fjg/src/downmem/build-rv/devApp/rvbins/LLAMA_GEMV_F32 \
GGML_DOWNMEM_NR_DPUS=4 \
GGML_DOWNMEM_MAX_COLS=128 \
GGML_DOWNMEM_VALIDATE=1 \
GGML_DOWNMEM_VERBOSE=1 \
./build-downmem/bin/test-downmem-backend
```

Expected:

```text
test-downmem-backend passed
```

- [ ] **Step 3: Run full E2E**

Run:

```bash
cd /home/fjg/src/llama.cpp
DOWNMEM_ROOT=/home/fjg/src/downmem \
MODEL=/home/fjg/src/llama.cpp/models/stories15M-q4_0.gguf \
N_PREDICT=1 \
PROMPT="Once upon a time" \
scripts/run-downmem-e2e.sh
```

Expected:

```text
==> E2E passed
```

- [ ] **Step 4: Confirm the offloaded op includes the new dimension**

Run:

```bash
cd /home/fjg/src/llama.cpp
grep -E "executed op=.*m=" build-downmem/downmem-e2e-logs/downmem.err
grep -E "executed op=.*m=" build-downmem/downmem-e2e-logs/ggml-test.log
```

Expected: both commands print at least one line.

- [ ] **Step 5: Run diff hygiene checks**

Run:

```bash
cd /home/fjg/src/llama.cpp
git diff --check
git status --short
cd /home/fjg/src/downmem
git diff --check -- devApp/LLAMA_GEMV_F32.c hostApp/LLAMA_GEMV_F32.c
git status --short
```

Expected: all `git diff --check` commands exit `0`. Status output shows only expected files changed plus pre-existing downmem untracked files.

## Acceptance Criteria

- Standalone Downmem `dmmLLAMA_GEMV_F32` passes for `m=1`, `m=3`, `m=4`, and `m=5` cases.
- `test-downmem-backend` compares CPU and Downmem for both GEMV and multi-column `GGML_OP_MUL_MAT`.
- `scripts/run-downmem-e2e.sh` completes with identical CPU and Downmem stdout.
- Downmem verbose logs include `m=` in claim and execution lines.
- Docs state that this extension is batched `MUL_MAT`, not general operator offload.
- RMS_NORM, ADD, MUL, GLU, ROPE, and FLASH_ATTN_EXT remain unsupported in code and documented as out of scope for this pass.

## Self-Review Notes

- Spec coverage: the plan reads the latest relevant llama.cpp and downmem files, extends a fixed compute-heavy operator, keeps the host and DPU split, and verifies full model inference equality.
- Scope: the plan avoids multi-binary runtime management and activation-only op scheduling changes.
- Type consistency: `m` is `uint32_t` in DPU args, host tests, and llama backend staging. `y` is stored globally as `[out_col * n + row]`; each DPU stores its local block as `[row * m + out_col]`.
- Repository constraints: no commit or push commands are included because the local `AGENTS.md` requires explicit human approval.
