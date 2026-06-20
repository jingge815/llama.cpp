#include "ggml-downmem.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

extern "C" {
#include <dpu.h>
}

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <string>
#include <vector>

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

static bool ggml_downmem_env_enabled(void) {
    const char * value = std::getenv("GGML_DOWNMEM");
    return value != nullptr && std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0;
}

static int ggml_downmem_env_i32(const char * name, int default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 0 || parsed > (1L << 30)) {
        return default_value;
    }
    return (int) parsed;
}

static const char * ggml_downmem_dpu_bin(void) {
    const char * value = std::getenv("GGML_DOWNMEM_DPU_BIN");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

static bool ggml_downmem_verbose(void) {
    const char * value = std::getenv("GGML_DOWNMEM_VERBOSE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static void ggml_downmem_verbose_log(const char * format, ...) {
    if (!ggml_downmem_verbose()) {
        return;
    }

    char message[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);

    std::fputs(message, stderr);
    std::fflush(stderr);
}

static bool ggml_downmem_allow_quant_dequant(void) {
    const char * value = std::getenv("GGML_DOWNMEM_ALLOW_QUANT_DEQUANT");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static bool ggml_downmem_file_exists(const char * path) {
    if (path == nullptr) {
        return false;
    }

    FILE * file = std::fopen(path, "rb");
    if (file == nullptr) {
        return false;
    }
    std::fclose(file);
    return true;
}

static bool ggml_downmem_src1_is_plain_f32_matrix(const ggml_tensor * t) {
    return t != nullptr &&
           t->type == GGML_TYPE_F32 &&
           t->nb[0] == (int64_t) sizeof(float) &&
           t->nb[1] >= t->nb[0] * t->ne[0] &&
           t->ne[2] == 1 &&
           t->ne[3] == 1;
}

static bool ggml_downmem_src0_can_stage_to_f32(const ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }

    if (t->type == GGML_TYPE_F32) {
        return t->nb[0] == (int64_t) sizeof(float) &&
               t->nb[1] >= t->nb[0] * t->ne[0];
    }

    if (!ggml_downmem_allow_quant_dequant()) {
        return false;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    return traits != nullptr && traits->to_float != nullptr;
}

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

static uint32_t ggml_downmem_align_up_u32(uint32_t v, uint32_t a) {
    return ((v + a - 1u) / a) * a;
}

static bool ggml_downmem_load(ggml_backend_downmem_context * ctx) {
    if (ctx->loaded) {
        return true;
    }
    if (ctx->failed) {
        return false;
    }

    const char * bin = ggml_downmem_dpu_bin();
    if (bin == nullptr || !ggml_downmem_file_exists(bin)) {
        GGML_LOG_ERROR("%s: invalid GGML_DOWNMEM_DPU_BIN\n", __func__);
        ctx->failed = true;
        return false;
    }

    ctx->nr_dpus = (uint32_t) ggml_downmem_env_i32("GGML_DOWNMEM_NR_DPUS", 4);
    if (ctx->nr_dpus == 0) {
        ctx->nr_dpus = 1;
    }
    ctx->binary_path = bin;

    if (dpu_alloc(ctx->nr_dpus, nullptr, &ctx->set) != DPU_OK) {
        GGML_LOG_ERROR("%s: dpu_alloc failed for %u DPUs\n", __func__, ctx->nr_dpus);
        ctx->failed = true;
        return false;
    }
    if (dpu_load(ctx->set, ctx->binary_path.c_str(), nullptr) != DPU_OK) {
        GGML_LOG_ERROR("%s: dpu_load failed for %s\n", __func__, ctx->binary_path.c_str());
        dpu_free(ctx->set);
        ctx->failed = true;
        return false;
    }

    uint32_t actual = 0;
    if (dpu_get_nr_dpus(ctx->set, &actual) != DPU_OK || actual == 0) {
        GGML_LOG_ERROR("%s: dpu_get_nr_dpus failed\n", __func__);
        dpu_free(ctx->set);
        ctx->failed = true;
        return false;
    }

    ctx->nr_dpus = actual;
    ctx->loaded = true;
    return true;
}

static bool ggml_downmem_stage_src0_row_major(const ggml_tensor * src0, float * a_rows, uint32_t k, uint32_t n, uint32_t k_pad) {
    if (src0->type == GGML_TYPE_F32) {
        for (uint32_t row = 0; row < n; ++row) {
            const char * src_row = (const char *) src0->data + row * src0->nb[1];
            std::memcpy(a_rows + (size_t) row * k_pad, src_row, k * sizeof(float));
        }
        return true;
    }

    if (!ggml_downmem_allow_quant_dequant()) {
        return false;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(src0->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        return false;
    }

    for (uint32_t row = 0; row < n; ++row) {
        const char * src_row = (const char *) src0->data + row * src0->nb[1];
        traits->to_float(src_row, a_rows + (size_t) row * k_pad, k);
    }
    return true;
}

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

static bool ggml_downmem_mul_mat_f32(ggml_backend_downmem_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!ggml_downmem_supports_mul_mat_f32(dst, true)) {
        GGML_LOG_ERROR("%s: scheduled unsupported op\n", __func__);
        return false;
    }
    if (!ggml_downmem_load(ctx)) {
        return false;
    }

    const uint32_t k = (uint32_t) src0->ne[0];
    const uint32_t n = (uint32_t) src0->ne[1];
    const uint32_t m = (uint32_t) src1->ne[1];
    const uint32_t k_pad = ggml_downmem_align_up_u32(k, 2);

    std::vector<dpu_rows_t> layout(ctx->nr_dpus);
    uint32_t max_rows = 0;
    for (uint32_t d = 0; d < ctx->nr_dpus; ++d) {
        const uint32_t base = n / ctx->nr_dpus;
        const uint32_t rest = n % ctx->nr_dpus;
        layout[d].rows = base + (d < rest ? 1u : 0u);
        layout[d].row_offset = d * base + (d < rest ? d : rest);
        max_rows = std::max(max_rows, layout[d].rows);
    }
    max_rows = std::max(max_rows, 1u);

    const uint32_t a_floats_per_dpu = max_rows * k_pad;
    const uint32_t x_floats_per_dpu = k_pad * m;
    const uint32_t y_floats_per_dpu = max_rows * m;
    const uint32_t a_bytes = a_floats_per_dpu * sizeof(float);
    const uint32_t x_bytes = x_floats_per_dpu * sizeof(float);
    const uint32_t y_bytes = y_floats_per_dpu * sizeof(float);

    const uint64_t total_mram_bytes = (uint64_t) a_bytes + x_bytes + y_bytes;
    if (total_mram_bytes > 64ull * 1024ull * 1024ull) {
        GGML_LOG_ERROR("%s: MRAM layout too large: %" PRIu64 " bytes\n", __func__, total_mram_bytes);
        return false;
    }

    std::vector<float> a_rows((size_t) n * k_pad, 0.0f);
    if (!ggml_downmem_stage_src0_row_major(src0, a_rows.data(), k, n, k_pad)) {
        GGML_LOG_ERROR("%s: failed to stage src0 type %s\n", __func__, ggml_type_name(src0->type));
        return false;
    }

    std::vector<float> a_padded((size_t) ctx->nr_dpus * a_floats_per_dpu, 0.0f);
    std::vector<float> x(x_floats_per_dpu, 0.0f);
    std::vector<float> y_padded((size_t) ctx->nr_dpus * y_floats_per_dpu, 0.0f);
    std::vector<float> y((size_t) n * m, 0.0f);
    std::vector<llama_gemv_f32_args_t> args(ctx->nr_dpus);

    ggml_downmem_stage_src1_matrix(src1, x.data(), k, m, k_pad);

    for (uint32_t d = 0; d < ctx->nr_dpus; ++d) {
        for (uint32_t row = 0; row < layout[d].rows; ++row) {
            const uint32_t global_row = layout[d].row_offset + row;
            std::memcpy(a_padded.data() + (size_t) d * a_floats_per_dpu + row * k_pad,
                        a_rows.data() + (size_t) global_row * k_pad,
                        k_pad * sizeof(float));
        }

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
    }

    dpu_set_t each;
    uint32_t idx = 0;
    DPU_FOREACH(ctx->set, each, idx) {
        if (dpu_prepare_xfer(each, args.data() + idx) != DPU_OK) {
            GGML_LOG_ERROR("%s: dpu_prepare_xfer args failed\n", __func__);
            return false;
        }
    }
    if (dpu_push_xfer(ctx->set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                      sizeof(llama_gemv_f32_args_t), DPU_XFER_DEFAULT) != DPU_OK) {
        GGML_LOG_ERROR("%s: args transfer failed\n", __func__);
        return false;
    }

    idx = 0;
    DPU_FOREACH(ctx->set, each, idx) {
        if (dpu_prepare_xfer(each, a_padded.data() + (size_t) idx * a_floats_per_dpu) != DPU_OK) {
            GGML_LOG_ERROR("%s: dpu_prepare_xfer A failed\n", __func__);
            return false;
        }
    }
    if (dpu_push_xfer(ctx->set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
                      a_bytes, DPU_XFER_DEFAULT) != DPU_OK) {
        GGML_LOG_ERROR("%s: A transfer failed\n", __func__);
        return false;
    }

    if (dpu_broadcast_to(ctx->set, DPU_MRAM_HEAP_POINTER_NAME, a_bytes,
                         x.data(), x_bytes, DPU_XFER_DEFAULT) != DPU_OK) {
        GGML_LOG_ERROR("%s: x transfer failed\n", __func__);
        return false;
    }

    if (dpu_launch(ctx->set, DPU_SYNCHRONOUS) != DPU_OK) {
        GGML_LOG_ERROR("%s: dpu_launch failed\n", __func__);
        return false;
    }

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

    return true;
}

static const char * ggml_backend_downmem_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "Downmem";
}

static void ggml_backend_downmem_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_downmem_context *) backend->context;
    if (ctx->loaded) {
        dpu_free(ctx->set);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_downmem_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_downmem_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (!ggml_downmem_mul_mat_f32(ctx, node)) {
                    return GGML_STATUS_FAILED;
                }
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_LOG_ERROR("%s: unsupported scheduled op %s\n", __func__, ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_downmem_i = {
    /* .get_name                = */ ggml_backend_downmem_get_name,
    /* .free                    = */ ggml_backend_downmem_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_downmem_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_downmem_guid(void) {
    static ggml_guid guid = { 0x44, 0x4d, 0x4d, 0x00, 0x6c, 0x6c, 0x61, 0x6d,
                              0x61, 0x2d, 0x64, 0x6f, 0x77, 0x6e, 0x6d, 0x01 };
    return &guid;
}

static ggml_backend_t ggml_backend_downmem_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_downmem_guid(),
        /* .iface     = */ ggml_backend_downmem_i,
        /* .device    = */ dev,
        /* .context   = */ new ggml_backend_downmem_context,
    };

    return backend;
}

static const char * ggml_backend_downmem_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Downmem";
}

static const char * ggml_backend_downmem_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Downmem RV simulator";
}

static void ggml_backend_downmem_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 64ull * 1024ull * 1024ull;
    *total = 64ull * 1024ull * 1024ull;
}

static enum ggml_backend_dev_type ggml_backend_downmem_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_downmem_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_downmem_device_get_name(dev);
    props->description = ggml_backend_downmem_device_get_description(dev);
    props->type = ggml_backend_downmem_device_get_type(dev);
    props->device_id = nullptr;
    ggml_backend_downmem_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_downmem_device_init(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_downmem_init(dev, params);
}

static ggml_backend_buffer_type_t ggml_backend_downmem_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static bool ggml_backend_downmem_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_downmem_supports_mul_mat_f32(op, false);
}

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

static bool ggml_backend_downmem_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i ggml_backend_downmem_device_i = {
    /* .get_name             = */ ggml_backend_downmem_device_get_name,
    /* .get_description      = */ ggml_backend_downmem_device_get_description,
    /* .get_memory           = */ ggml_backend_downmem_device_get_memory,
    /* .get_type             = */ ggml_backend_downmem_device_get_type,
    /* .get_props            = */ ggml_backend_downmem_device_get_props,
    /* .init_backend         = */ ggml_backend_downmem_device_init,
    /* .get_buffer_type      = */ ggml_backend_downmem_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_downmem_device_supports_op,
    /* .supports_buft        = */ ggml_backend_downmem_device_supports_buft,
    /* .offload_op           = */ ggml_backend_downmem_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

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
