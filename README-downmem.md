# Downmem 适配说明

## 仓库用途

这个 fork 主要用于在 `llama.cpp` 中开发和验证 Downmem 适配. 当前重点是把
ggml 中窄范围 F32 `GGML_OP_MUL_MAT` 和 opt-in `GGML_OP_SCALE` offload 到
Downmem RV simulator, 用于验证 Downmem runtime, DPU binary 和 llama.cpp
后端调度之间的端到端链路.

上游仓库仍然是:

```text
https://github.com/ggml-org/llama.cpp.git
```

个人 fork 远端是:

```text
https://github.com/jingge815/llama.cpp.git
```

当前开发分支默认使用:

```text
feature/llama-downmem-offload
```

## 功能简述

当前 Downmem 适配包含以下内容:

- 新增 `ggml-downmem` 后端, 通过 `dpu.h` 调用 Downmem runtime.
- 注册 `Downmem` ggml backend device, 可通过 `--device Downmem` 选择.
- 支持窄范围 F32 `GGML_OP_MUL_MAT`, 包括单列 GEMV 和少量 RHS 列的 batched
  MUL_MAT.
- 支持 opt-in 窄范围 F32 `GGML_OP_SCALE`, 用于验证多算子 Downmem 链路.
- 支持 F32 矩阵输入; 量化矩阵可在 host 端 dequant 到 F32 后 offload.
- 支持通过环境变量控制 DPU binary, DPU 数量, 最大 offload op 数量和校验模式.
- 新增 `test-downmem-backend` 正确性测试.
- 新增 `scripts/downmem-smoke.sh` 短流程验证脚本.
- 新增 `scripts/run-downmem-e2e.sh` 一键端到端验证脚本.
- 新增中文技术文档: `docs/backend/DOWNMEM.md`.

当前实现不是完整通用加速后端. 它只覆盖矩阵乘和 SCALE 中的窄范围 F32
路径, 主要用于 Downmem 适配验证和迭代.

## 使用方法

### 1. 确认远端和分支

```sh
git remote -v
git branch --show-current
```

预期远端包含:

```text
origin    https://github.com/ggml-org/llama.cpp.git
jingge815 https://github.com/jingge815/llama.cpp.git
```

预期开发分支:

```text
feature/llama-downmem-offload
```

如果没有个人 fork 远端, 添加:

```sh
git remote add jingge815 https://github.com/jingge815/llama.cpp.git
```

### 2. 构建 Downmem 版本

```sh
cmake -S . -B build-downmem \
  -DGGML_DOWNMEM=ON \
  -DGGML_DOWNMEM_ROOT=/home/fjg/src/downmem \
  -DGGML_DOWNMEM_LIB=/home/fjg/src/downmem/build-rv/libdmmShared.so \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-downmem --target test-downmem-backend llama-completion -j"$(nproc)"
```

### 3. 运行一键 E2E 验证

```sh
DOWNMEM_ROOT=/home/fjg/src/downmem \
MODEL=/home/fjg/src/llama.cpp/models/stories15M-q4_0.gguf \
scripts/run-downmem-e2e.sh
```

脚本会依次完成:

- 构建 Downmem runtime 和 `LLAMA_GEMV_F32` DPU binary.
- 配置并构建 llama.cpp Downmem 后端.
- 运行 standalone Downmem `LLAMA_GEMV_F32` 测试.
- 运行 ggml 层 `test-downmem-backend`, 覆盖 MUL_MAT 和 SCALE.
- 分别运行 CPU 和 Downmem completion, 并要求 Downmem completion 日志同时出现
  MUL_MAT 和 SCALE.
- 比较 CPU 与 Downmem 的 stdout.

默认日志目录:

```text
build-downmem/downmem-e2e-logs
```

### 4. 手动运行 completion

```sh
DMM_NR_SIM_THRDS=4 \
GGML_DOWNMEM=1 \
GGML_DOWNMEM_DPU_BIN=/home/fjg/src/downmem/build-rv/devApp/rvbins/LLAMA_GEMV_F32 \
GGML_DOWNMEM_NR_DPUS=4 \
GGML_DOWNMEM_MAX_OPS=1 \
GGML_DOWNMEM_MAX_SCALE_OPS=2 \
GGML_DOWNMEM_MAX_COLS=128 \
GGML_DOWNMEM_ENABLE_SCALE=1 \
GGML_DOWNMEM_ALLOW_QUANT_DEQUANT=1 \
GGML_DOWNMEM_VERBOSE=1 \
build-downmem/bin/llama-completion \
  -m models/stories15M-q4_0.gguf \
  -p "Once upon a time" \
  -n 1 \
  -t 1 -tb 1 \
  -s 42 \
  --temp 0.8 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 \
  --backend-sampling \
  --no-display-prompt \
  --no-warmup \
  --no-context-shift \
  --device Downmem \
  -fit off \
  --log-verbosity 3
```

预期日志中包含:

```text
claiming MUL_MAT
executed op=1 ... m=1
kind=SCALE
```

更多技术细节见:

```text
docs/backend/DOWNMEM.md
```

## 提交方法

开发时尽量保持提交范围小而清楚. 建议流程:

```sh
git status --short
git diff
git add <changed-files>
git commit -m "module : concise change summary"
```

如果提交由 AI 工具辅助完成, 按仓库规则在提交信息正文加入:

```text
Assisted-by: Codex
```

示例:

```sh
git commit -m "docs : update Downmem usage notes" -m "Assisted-by: Codex"
```

注意:

- 不要把临时 patch, build 产物, 日志目录或模型文件提交进仓库.
- 提交前用 `git status --short` 确认 staged 文件范围.
- 涉及后端行为时, 至少运行 `test-downmem-backend` 或
  `scripts/run-downmem-e2e.sh`.
- 面向上游 PR 前, 需要自己确认能解释每一处改动, 并遵守 upstream
  `AGENTS.md` 和 `CONTRIBUTING.md`.

## 推送方法

本 fork 的默认推送目标是个人 fork 上的当前开发分支:

```sh
git push -u jingge815 feature/llama-downmem-offload
```

后续已经设置 upstream 后, 可以使用:

```sh
git push
```

只推送到个人 fork. 不要直接推送到 `origin`, 因为 `origin` 是上游
`ggml-org/llama.cpp`.

## 同步上游

需要同步 upstream 时:

```sh
git fetch origin
git rebase origin/master
```

如果发生冲突, 先解决冲突并重新运行相关验证, 再推送到个人 fork.

## 常用检查命令

```sh
git status --short --branch
git remote -v
git log --oneline --decorate -5
git diff --stat
```
