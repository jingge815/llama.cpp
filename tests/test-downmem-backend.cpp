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

static float pattern_x(int col) {
    const int v = (col * 19 + 5) % 29 - 14;
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

static void fill_vector_f32(ggml_tensor * t, int cols) {
    float * dst = (float *) t->data;
    for (int col = 0; col < cols; ++col) {
        dst[col] = pattern_x(col);
    }
}

static std::vector<float> run_graph(ggml_backend_t backend, int n, int k) {
    const size_t mem_size = 16u * 1024u * 1024u;
    ggml_init_params params = {
        /* .mem_size   = */ mem_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };

    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    fill_tensor_2d_f32(a, n, k);
    fill_vector_f32(x, k);

    ggml_tensor * y = ggml_mul_mat(ctx, a, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed: %d\n", (int) status);
        std::exit(1);
    }

    std::vector<float> result(n);
    for (int row = 0; row < n; ++row) {
        result[row] = *(float *) ((char *) y->data + row * y->nb[0]);
    }

    ggml_free(ctx);
    return result;
}

int main(void) {
    const char * enabled = std::getenv("GGML_DOWNMEM");
    const char * bin = std::getenv("GGML_DOWNMEM_DPU_BIN");
    if (enabled == nullptr || std::strcmp(enabled, "0") == 0 || bin == nullptr || bin[0] == '\0') {
        std::puts("SKIP test-downmem-backend: GGML_DOWNMEM and GGML_DOWNMEM_DPU_BIN are required");
        return 77;
    }

    const int n = 17;
    const int k = 31;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        fprintf(stderr, "failed to init CPU backend\n");
        return 1;
    }
    std::vector<float> expected = run_graph(cpu, n, k);
    ggml_backend_free(cpu);

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Downmem");
    if (dev == nullptr) {
        fprintf(stderr, "Downmem backend device not registered\n");
        return 1;
    }

    ggml_backend_t downmem = ggml_backend_dev_init(dev, nullptr);
    if (downmem == nullptr) {
        fprintf(stderr, "failed to init Downmem backend\n");
        return 1;
    }
    std::vector<float> got = run_graph(downmem, n, k);
    ggml_backend_free(downmem);

    for (int i = 0; i < n; ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float scale = std::fmax(1.0f, std::fabs(expected[i]));
        if (!(diff <= 1.0e-4f || diff <= scale * 1.0e-4f)) {
            fprintf(stderr, "row=%d expected=%g got=%g diff=%g\n",
                    i, (double) expected[i], (double) got[i], (double) diff);
            return 1;
        }
    }

    std::puts("test-downmem-backend passed");
    return 0;
}
