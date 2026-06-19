# llama.cpp Downmem 后端

## 概览

这两份 Downmem patch 为 ggml 增加了一个实验性后端. 该后端通过
`dpu.h` 运行时把一个很窄的 GEMV 场景 offload 到 Downmem RV simulator.
它主要面向 LLM 逐 token 生成路径, 因为生成阶段很多层会退化为矩阵乘以单个
激活向量.

两份 patch 的职责如下:

- `0001-Add-downmem-ggml-GEMV-offload-backend.patch`: 新增 ggml 后端,
  CMake 集成, 后端正确性测试, 以及一个 smoke 脚本.
- `0002-Add-one-command-downmem-E2E-runner.patch`: 新增一键端到端脚本,
  用于构建 Downmem 组件, 构建 llama.cpp, 验证后端, 运行 CPU 和 Downmem
  completion, 并比较生成 stdout.

这个后端不是通用加速后端. 它当前只支持符合 GEMV 形状的
`GGML_OP_MUL_MAT`, 并且输出必须是 F32.

## 修改内容

后端 patch 增加了以下 llama.cpp 集成点:

- `ggml/CMakeLists.txt`: 增加 `GGML_DOWNMEM`, `GGML_DOWNMEM_ROOT`,
  `GGML_DOWNMEM_LIB`.
- `ggml/src/CMakeLists.txt`: 增加 `ggml_add_backend(Downmem)`.
- `ggml/src/ggml-backend-reg.cpp`: 在启用 `GGML_USE_DOWNMEM` 时注册后端.
- `ggml/src/ggml-downmem/`: 新增后端实现和注册头文件.
- `tests/test-downmem-backend.cpp`: 构造一个小型 F32 GEMV 图, 将结果与 CPU
  后端比较.
- `tests/CMakeLists.txt`: 在启用 Downmem 后端时构建并注册测试.
- `scripts/downmem-smoke.sh`: 构建并运行短流程 smoke 测试.
- `scripts/run-downmem-e2e.sh`: 构建并运行完整端到端验证流程.

## 编译选项

| 选项 | 默认值 | 作用 |
| --- | --- | --- |
| `GGML_DOWNMEM` | `OFF` | 启用 Downmem ggml 后端. |
| `GGML_DOWNMEM_ROOT` | `/home/fjg/src/downmem` | Downmem 源码树路径. |
| `GGML_DOWNMEM_LIB` | `${GGML_DOWNMEM_ROOT}/build-rv/libdmmShared.so` | Downmem shared runtime library 路径. |

示例:

```sh
cmake -S . -B build-downmem \
  -DGGML_DOWNMEM=ON \
  -DGGML_DOWNMEM_ROOT=/home/fjg/src/downmem \
  -DGGML_DOWNMEM_LIB=/home/fjg/src/downmem/build-rv/libdmmShared.so \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-downmem --target test-downmem-backend llama-completion -j"$(nproc)"
```

## 运行时变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `GGML_DOWNMEM` | 未设置 | 必须设为非零且不是 `false`, 后端才会 claim 工作. |
| `GGML_DOWNMEM_DPU_BIN` | 未设置 | DPU binary 路径, 通常是 `build-rv/devApp/rvbins/LLAMA_GEMV_F32`. |
| `GGML_DOWNMEM_NR_DPUS` | `4` | 向运行时申请的 DPU 数量. |
| `GGML_DOWNMEM_MAX_OPS` | 很大的默认值 | 最多 claim 多少个符合条件的 op. completion 脚本默认设为 `1`. |
| `GGML_DOWNMEM_ALLOW_QUANT_DEQUANT` | 未设置 | 允许把非 F32 矩阵 tensor 在 host 端 dequant 到 F32 后再 offload. |
| `GGML_DOWNMEM_VALIDATE` | 未设置 | 每次 offload 后在 host 端重算结果, 并与 DPU 结果比较. |
| `GGML_DOWNMEM_VERBOSE` | 未设置 | 向 stderr 打印 claimed 和 executed op 日志. |
| `DMM_NR_SIM_THRDS` | 脚本默认 `4` | 控制 Downmem simulator 线程数. |

## 支持的算子

后端只支持满足以下条件的 `GGML_OP_MUL_MAT`:

- `src0` 是 2D 矩阵, 且 `ne[2] == ne[3] == 1`.
- `src1` 是一个 F32 向量, 以单行 tensor 表示.
- `dst` 是 F32, 且只有一个输出列.
- `src0->ne[0] == src1->ne[0]`.
- `dst->ne[0] == src0->ne[1]`, 且 `dst->ne[1] == 1`.
- `src1` 和 `dst` 的布局必须能被后端按连续 F32 数据 staging.
- `src0` 要么本身是 F32, 要么可以通过
  `ggml_get_type_traits(type)->to_float` 转成 F32. 量化输入需要设置
  `GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1`.

后端声明使用 host buffer type, 不支持 async 或 event capability. 它不会在
simulator 侧长期持有模型内存, 而是对每个 claimed op 从 host memory 临时
staging 数据.

## 执行原理

每个被 claim 的 GEMV op 会按以下流程执行:

1. 检查 `GGML_DOWNMEM`. 当需要运行时就绪时, 检查 `GGML_DOWNMEM_DPU_BIN`.
   DPU set 会 lazy allocate 和 lazy load.
2. 读取 `GGML_DOWNMEM_NR_DPUS`, 通过 `dpu_alloc` 申请 DPU, 再通过
   `dpu_load` 加载 `LLAMA_GEMV_F32` binary.
3. 将矩阵行尽量平均分配到多个 DPU.
4. 将 `src0` staging 成 row-major F32 host buffer. 如果是量化矩阵且设置了
   `GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1`, 则先在 host 端 dequant.
5. 从 `src1` staging F32 向量.
6. 为每个 DPU 构造参数:
   - `k`: 输入宽度.
   - `k_pad`: 将 `k` 向上对齐到偶数.
   - `n_rows`: 分配给当前 DPU 的行数.
   - `max_rows`: 任意 DPU 上的最大行数.
   - `a_offset`, `x_offset`, `y_offset`: MRAM 布局偏移.
7. 把参数和矩阵行传到每个 DPU, 把向量 broadcast 到所有 DPU, 同步 launch,
   然后读回输出行.
8. 如果设置了 `GGML_DOWNMEM_VALIDATE=1`, 则在 host 端用 F32 dot product
   重算并校验结果.
9. 将 F32 输出写回 `dst`.

每个 DPU 的 MRAM 布局如下:

```text
[ A rows ][ x vector ][ y rows ]
```

如果单个 DPU 的 staging 布局超过 64 MiB, 后端会拒绝执行该 op.

## 使用方法

### 一键 E2E 流程

需要验证完整链路时, 使用 E2E runner:

```sh
DOWNMEM_ROOT=/home/fjg/src/downmem \
MODEL=/home/fjg/src/llama.cpp/models/stories15M-q4_0.gguf \
scripts/run-downmem-e2e.sh
```

脚本执行以下步骤:

1. 检查 GGUF 模型和 `libdmmShared.so` 是否存在.
2. 构建 Downmem targets: `dmmShared`, `rvLLAMA_GEMV_F32`,
   `dmmLLAMA_GEMV_F32`.
3. 运行独立 Downmem `LLAMA_GEMV_F32` 测试, 并要求日志中出现
   `LLAMA_GEMV_F32 passed`.
4. 使用 `GGML_DOWNMEM=ON` 配置 llama.cpp.
5. 构建 `test-downmem-backend` 和 `llama-completion`.
6. 运行 ggml 层 Downmem 正确性测试, 并要求同时出现
   `test-downmem-backend passed` 和 `executed op=1`.
7. 运行确定性的 CPU completion.
8. 使用 `--device Downmem` 运行确定性的 Downmem completion.
9. 检查 offload 日志, 并比较 CPU stdout 与 Downmem stdout.

默认日志目录:

```text
build-downmem/downmem-e2e-logs
```

常用覆盖变量:

```sh
DOWNMEM_ROOT=/path/to/downmem
DOWNMEM_BUILD=/path/to/downmem/build-rv
LLAMA_BUILD=/path/to/llama-build
GGML_DOWNMEM_DPU_BIN=/path/to/LLAMA_GEMV_F32
MODEL=/path/to/model.gguf
N_PREDICT=1
PROMPT="Once upon a time"
LOG_DIR=/tmp/downmem-e2e
scripts/run-downmem-e2e.sh
```

### Smoke 流程

需要较短验证流程时, 使用 smoke 脚本:

```sh
DOWNMEM_ROOT=/home/fjg/src/downmem \
MODEL=/home/fjg/src/llama.cpp/models/stories15M-q4_0.gguf \
scripts/downmem-smoke.sh
```

它会构建 DPU kernel 和 llama.cpp targets, 运行后端单元测试, 然后用
`--device Downmem` 运行一次 `llama-completion`.

### 手动 completion 运行

构建完成后, 可以手动运行 Downmem completion:

```sh
DMM_NR_SIM_THRDS=4 \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN=/home/fjg/src/downmem/build-rv/devApp/rvbins/LLAMA_GEMV_F32 \
GGML_DOWNMEM_NR_DPUS=4 \
GGML_DOWNMEM_MAX_OPS=1 \
GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1 \
GGML_DOWNMEM_VERBOSE=1 \
build-downmem/bin/llama-completion \
  -m models/stories15M-q4_0.gguf \
  -p "Once upon a time" \
  -n 1 \
  -t 1 -tb 1 \
  -s 42 \
  --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
  --no-display-prompt \
  --no-warmup \
  --no-context-shift \
  --device Downmem \
  -fit off \
  --log-verbosity 3
```

stderr 中预期会出现类似日志:

```text
ggml_backend_downmem_device_offload_op: claiming MUL_MAT ...
ggml_downmem_mul_mat_gemv: executed op=1 ...
```

## 正确性验证

后端有两层正确性验证:

- `test-downmem-backend`: 构造一个 `n = 17`, `k = 31` 的小型 F32 图,
  先在 CPU 后端计算, 再在 Downmem 后端计算, 然后用 `1e-4` 绝对误差或相对
  误差阈值逐行比较. 如果没有设置 `GGML_DOWNMEM` 或 `GGML_DOWNMEM_DPU_BIN`,
  测试返回 `77`, CTest 会把它视为 skipped.
- `GGML_DOWNMEM_VALIDATE=1`: 后端内部会对每个 offloaded op 在 host 端重算,
  如果任意行超过同样的误差阈值, graph compute 失败.

E2E runner 还会在应用层比较确定性 CPU completion 和 Downmem completion
的 stdout.

## 当前效果

默认 E2E 设置下, 只 offload 一个符合条件的 GEMV op:

```sh
GGML_DOWNMEM_MAX_OPS=1
```

这样便于检查: completion 输出应与 CPU 一致, 日志中应只出现一个
claimed/executed Downmem op. 增大 `GGML_DOWNMEM_MAX_OPS` 可以让更多符合条件
的 GEMV op 被 claim, 但当前实现仍然会对每个 op 经由 host memory staging,
并同步 launch DPU.

## 限制

- 后端依赖外部 Downmem 源码树, runtime library 和 DPU binary. 它不是
  llama.cpp 内部自包含组件.
- 默认路径是本机开发路径 `/home/fjg/src`.
- 只包含 Linux 相关 RPATH 处理.
- 只实现了 GEMV 形状的 `GGML_OP_MUL_MAT`.
- 量化矩阵不会以量化形式直接计算, 而是先在 host 端 dequant 到 F32 再
  offload.
- 后端使用 host buffers 和同步执行.
- 当前脚本默认使用 `llama-completion` 和一个本地小模型.
- 性能数据应理解为 simulator/offload plumbing 验证结果, 除非在有代表性的
  Downmem 配置上重新测量.

## 故障排查

- `Downmem backend device not registered`: 重新使用 `-DGGML_DOWNMEM=ON`
  构建, 并确认 `ggml-downmem` 已构建.
- `invalid GGML_DOWNMEM_DPU_BIN`: 将 `GGML_DOWNMEM_DPU_BIN` 设为存在的
  `LLAMA_GEMV_F32` binary.
- `missing downmem shared library`: 构建 Downmem runtime, 或把
  `GGML_DOWNMEM_LIB` 和 `DOWNMEM_BUILD` 设为正确路径.
- 没有 `claiming MUL_MAT` 日志: 设置 `GGML_DOWNMEM=1`, 设置有效的
  `GGML_DOWNMEM_DPU_BIN`, 传入 `--device Downmem`, 启用
  `GGML_DOWNMEM_VERBOSE=1`, 并检查 op 形状是否满足支持的 GEMV 条件.
- E2E runner 中 CPU 和 Downmem stdout 不一致: 查看 `LOG_DIR` 下的
  `cpu-vs-downmem.diff`, `cpu.err`, `downmem.err`.
