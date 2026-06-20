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
