# experiments/ — 实验脚本与原始日志归档

这里是三条主线（sm89 Triton 算子 / 服务层瓶颈归因 / vLLM 横向对标）**真正跑过的命令**。
`docs/` 里的 worklog 负责讲结论与错误账本，本目录只管**怎么重跑**——数字要能复现，
所以脚本按原样保留，只做了「从工作区临时目录搬进版本控制」和「按主线分组」两件事。

> 背景：这些脚本原先散落在被 gitignore 的 `.tmp/` 里。它们不会被 import，
> 也不会被任何包引用，纯粹是一批「记录实验过程的一次性脚本」，所以刻意放在仓库根下的
> `experiments/`，而不是混进 `benchmarks/`（后者是**交付给使用者**的、有 CLI 的基准工具）。

## 目录结构

```
experiments/
├── shared/    17 个：Windows↔WSL 搬运与环境探测（所有实验的公共前置）
├── sm89/      14 个：Triton decode attention 的开发、扫描、取证
├── serving/   15 个：服务层瓶颈归因（max_workers / step 剖析 / 过载边界）
├── vllm/      27 个：与 vLLM 0.31.0 的同口径对标（数据集、KV 探测、压测、nsys 三连）
└── logs/       6 个：当时落盘的标准输出（几万行原始日志，仅存档）
```

worklog 里提到脚本时只写文件名（例如 `sync_to_wsl.sh`、`run_cmp3.sh`），
按名字在上表对应子目录里找即可。

> ⚠️ **一处缺口**：`run_cmp4.sh` 只留下了输出日志 `logs/run_cmp4.out`，脚本本体没能带进版本控制
> （当时直接在 WSL 侧改的，Windows 侧已经没有了）。其余 worklog 提到的脚本都在。

## 执行环境（重要）

代码写在 **Windows 仓库**，但只有 WSL 里有可用工具链
（torch 2.13.0+cu130 / triton 3.7.1 / flash_attn 2.8.3.post1 / RTX 4060 sm_89），
所以流程固定是「Windows 写 → rsync 到 WSL → 在 WSL 里跑」：

```bash
# 1) 同步（不带 --delete，不动 WSL 独有文件）
bash experiments/shared/sync_to_wsl.sh

# 2) 在 WSL 里跑指定脚本（一律用脚本文件，不要在 pwsh 里内联引号）
wsl.exe -d Ubuntu -- bash /mnt/c/.../experiments/sm89/run_decode_bench.sh
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml
```

三条踩坑记录（worklog 里有详细版）：**不要在 pwsh 里内联 Python 引号**（会被转义搞坏，
一律写脚本文件）；`pkill -f '<pattern>'` 会匹配到自身命令行（用 `'run_cmp[3].sh'` 方括号写法）；
`/tmp` 在两次 `wsl.exe` 调用之间可能被清空（中间产物写仓库目录）。

## shared/ — 环境与搬运

| 脚本 | 作用 |
|---|---|
| `sync_to_wsl.sh` | 仓库 → WSL 的单向同步（所有实验的第一步） |
| `pip_fix.py` | 受限环境里跑 pip 的 `mkdtemp` 补丁（`python pip_fix.py install ...`） |
| `wsl_probe.sh` / `wsl_probe2.sh` | 探 WSL 侧 CUDA / 包版本 |
| `check_nvidia_repo.sh` / `fix_nsight_install.sh` | 查 NVIDIA 源、修 nsys/ncu 安装 |
| `probe_flash_attn_constraint.sh` / `probe_fa2.sh` | 探 flash-attn 的版本与 `% 256` 约束 |
| `probe_profiling.sh` / `probe_cmp_env.sh` | 探 profiler 可用性、对标环境一致性 |
| `introspect_vllm.sh` / `probe_vllm_api.sh` | 探 vLLM 0.31.0 的真实 API / CLI（文档与实现不一致处） |
| `vllm_defaults.sh` / `vllm_defaults2.sh` / `vllm_logflag.sh` | 查 vLLM 的默认值与日志开关 |
| `archive_paths.py` | 把 worklog 里的 `.tmp/` 路径改写到本目录（一次性，已执行） |
| `group_experiments.py` | 按主线把脚本分到 shared/sm89/serving/vllm（一次性，已执行） |

## sm89/ — Triton decode attention

对应 `docs/sm89_worklog.md`（开发过程）与 `docs/sm89_phase1_results.md`（交付结果）。

| 脚本 | 作用 |
|---|---|
| `run_decode_tests.sh` | 跑 `tests/test_decode_attention.py`（29 项） |
| `run_decode_bench.sh` | 跑 `benchmarks/benchmark_decode_attention.py` |
| `run_sweeps.sh` / `run_stages.sh` / `run_splits.sh` | block_n×num_warps / num_stages / split-KV 三组扫描 |
| `run_kernel_families.sh` | 逐 kernel 家族对比 |
| `run_attn_share.sh` | 量 attention 在整步里的真实占比（纠正「<3%」那次） |
| `run_roofline.sh` / `run_onchip_math.sh` / `run_gemm_sweep.sh` | roofline 常数、片上算力、GEMM occupancy 扫描 |
| `run_shortctx_cmp.sh` | 高 batch × 极短 ctx 下 flash-attn 耗时超线性的直接验证 |
| `run_ncu_verify.sh` | ncu 硬件计数器取证（`dram__bytes_read` / 吞吐） |
| `run_e2e.sh` / `run_final.sh` | 端到端回归与 Phase 1 最终验收（含 eager / CUDA graph 对照） |

## serving/ — 服务层瓶颈归因

对应 `docs/cmp_vllm_vs_nanovllm_worklog.md` 的「服务层税」一节，以及
`docs/sm89_bottleneck_analysis.md` 里的更正记录。

| 脚本 | 作用 |
|---|---|
| `run_maxworkers_test.sh` | `max_workers` 10 → 64 的对照压测（3.4× 的来源） |
| `run_instrumented_server.sh` | 用 `benchmarks/profile_serving_step.py` 起插桩服务 |
| `run_step_profile.sh` / `run_final_profile.sh` | 步级 host/gpu 时长 + 真实 batch 直方图 |
| `run_confirm_fix.sh` / `run_verify_fix.sh` | 验证修复（含 p99） |
| `run_boundary.sh` | 并发上限（KV 池硬顶 ≈162 序列，背压拒绝而非 OOM） |
| `run_online_bench.sh` / `run_concurrency_scale.sh` | 在线吞吐随并发扫描 |
| `run_p1.sh` / `run_p1b.sh` | 追「实测 2930 vs 引擎能力 5690 tok/s」之间剩下的 1.9× |
| `run_busy_analysis.sh` / `run_busy_fraction.sh` / `run_busy_oneshot.sh` | GPU 忙占比三种测法（含「反推代替直测」的更正） |
| `stop_and_report.sh` | 停残留进程并落盘 |

## vllm/ — 与 vLLM 0.31.0 的同口径对标

对应 `docs/cmp_vllm_vs_nanovllm_worklog.md`。

| 脚本 | 作用 |
|---|---|
| `probe_kv_size.py` / `probe_kv_max.sh` | 探**我们**的 KV 上限 |
| `probe_vllm_kv_max.sh` / `probe_vllm_kv_push.sh` | 探 **vLLM** 的 KV 上限与「顶满会怎样」 |
| `verify_kv_pressure.sh` | 验证 168 vs 144 blocks 的显存压力假设 |
| `run_cmp.sh` → `run_cmp3.sh` | 对标主流程的逐轮演进（`run_cmp3.sh` = 定案的 KV 对齐 + 重复 + GPU 采样） |
| `run_cmp_kernels.sh` / `run_cmp_offline_curve.sh` | 逐 kernel 对比、离线吞吐曲线 |
| `run_official.sh` + `check_official_dataset.py` / `check_official_scripts.sh` + `read_official.py` | 复现官方 `bench.py` 口径（133,966 tokens 那张表） |
| `run_nsys_steady.sh` / `run_nsys_nodes.sh` / `run_nsys_reports.sh` | 稳态取样的 nsys 三连（含 `--cuda-graph-trace=node` 与 `cudaProfilerApi` 夹取） |
| `diag_n8.py` / `diag_vllm_step.py` | 冷启动污染诊断（2627 ms → 167 ms 那次） |
| `smoke_cmp_offline.sh` / `smoke_cmp_online.sh` | 对标前冒烟 |
| `smoke_vllm.sh` / `smoke_vllm2.sh` / `smoke_profilers.sh` | vLLM 与 profiler 冒烟 |
| `read_results.py` / `summarize_results.py` / `verify_results.py` | 结果读取、汇总、交叉校验 |

## logs/

| 文件 | 内容 |
|---|---|
| `run_cmp.out` / `run_cmp2.out` / `run_cmp3.out` / `run_cmp4.out` | 对标逐轮原始输出（几万行，含中途失败的轮次） |
| `run_official.out` | 官方 `bench.py` 口径的原始输出 |
| `smoke_vllm2.out` | vLLM 冒烟输出 |

日志是**当时的原始产物**：里面的绝对路径、失败重试、被推翻的中间结论都保持原样，不做美化。
