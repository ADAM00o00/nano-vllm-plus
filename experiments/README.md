# experiments/ — 实验脚本与原始日志归档

这里是三条主线（sm89 Triton 算子 / 服务层瓶颈归因 / vLLM 横向对标）**真正跑过的命令**。
`docs/` 里的 worklog 负责讲结论与错误账本，本目录只管**怎么重跑**——数字要能复现，
所以脚本按原样保留，只做了「从工作区临时目录搬进版本控制」这一件事。

> 背景：这些脚本原先散落在被 gitignore 的 `.tmp/` 里。它们不会被 import，
> 也不会被任何包引用，纯粹是一批「记录实验过程的一次性脚本」，所以刻意放在仓库根下的
> `experiments/`，而不是混进 `benchmarks/`（后者是**交付给使用者**的、有 CLI 的基准工具）。

## 目录约定

| 目录 | 内容 |
|---|---|
| `scripts/` | 可直接执行的 `.sh` / `.py` 脚本（70 个） |
| `logs/`  | 当时落盘的标准输出（6 个 `.out`，几万行原始日志，仅存档） |

## 执行环境（重要）

代码写在 **Windows 仓库**，但只有 WSL 里有可用工具链
（torch 2.13.0+cu130 / triton 3.7.1 / flash_attn 2.8.3.post1 / RTX 4060 sm_89），
所以流程固定是「Windows 写 → rsync 到 WSL → 在 WSL 里跑」：

```bash
bash experiments/scripts/sync_to_wsl.sh                    # rsync，不带 --delete，不动 WSL 独有文件
wsl.exe -d Ubuntu -- bash /mnt/c/.../experiments/scripts/<script>.sh
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml
```

三条踩坑记录（worklog 里有详细版）：**不要在 pwsh 里内联 Python 引号**（会被转义搞坏，
一律写脚本文件）；`pkill -f '<pattern>'` 会匹配到自身命令行（用 `'run_cmp[3].sh'` 方括号写法）；
`/tmp` 在两次 `wsl.exe` 调用之间可能被清空（中间产物写仓库目录）。

## 关键脚本（要复现结论从这里入手）

### 环境 / 联通

| 脚本 | 作用 |
|---|---|
| `sync_to_wsl.sh` | 仓库 → WSL 的单向同步（所有实验的第一步） |
| `wsl_probe.sh` / `wsl_probe2.sh` | 探 WSL 侧 CUDA / 包版本 |
| `check_nvidia_repo.sh` | 查 NVIDIA 源（装 nsight 之前） |
| `fix_nsight_install.sh` | 修 nsys/ncu 安装 |
| `probe_flash_attn_constraint.sh` / `probe_fa2.sh` | 探 flash-attn 的版本与 `% 256` 约束 |

### sm89 Triton decode attention（对应 `docs/sm89_worklog.md`）

| 脚本 | 作用 |
|---|---|
| `run_decode_tests.sh` | 跑 `tests/test_decode_attention.py` |
| `run_decode_bench.sh` | 跑 `benchmarks/benchmark_decode_attention.py` |
| `run_sweeps.sh` / `run_stages.sh` / `run_splits.sh` | block_n×num_warps / num_stages / split-KV 三组扫描 |
| `run_kernel_families.sh` | 逐 kernel 家族对比 |
| `run_e2e.sh` | 端到端回归（含 CUDA graph 与 eager 对照） |
| `run_p1.sh` / `run_p1b.sh` | Phase 1 验收的一键跑批 |
| `run_ncu_verify.sh` | ncu 硬件计数器取证（DRAM 字节 / 吞吐） |
| `stop_and_report.sh` | 停残留进程并落盘 |

### 服务层瓶颈归因（对应 worklog 的 `max_workers` 一节）

| 脚本 | 作用 |
|---|---|
| `run_maxworkers_test.sh` | `max_workers` 10 → 64 的对照压测（3.4× 的来源） |
| `run_instrumented_server.sh` | 用 `benchmarks/profile_serving_step.py` 起插桩服务 |
| `run_step_profile.sh` / `run_final_profile.sh` | 步级 host/gpu 时长 + 真实 batch 直方图 |
| `run_confirm_fix.sh` / `run_verify_fix.sh` | 验证修复（含 p99） |
| `run_boundary.sh` | 并发上限（KV 池硬顶 ≈162 序列） |
| `run_online_bench.sh` / `run_concurrency_scale.sh` | 在线吞吐随并发扫描 |
| `run_busy_*.sh` | GPU 忙占比三种测法（含那次「反推代替直测」的更正） |

### vLLM 横向对标（对应 `docs/cmp_vllm_vs_nanovllm_worklog.md`）

| 脚本 | 作用 |
|---|---|
| `probe_kv_size.py` / `probe_kv_max.sh` | 探**我们**的 KV 上限 |
| `probe_vllm_kv_max.sh` / `probe_vllm_kv_push.sh` | 探 **vLLM** 的 KV 上限与「顶满会怎样」 |
| `verify_kv_pressure.sh` | 验证 168 vs 144 blocks 的显存压力假设 |
| `run_cmp.sh` → `run_cmp4.sh` | 对标主流程的逐轮演进（`run_cmp3.sh` = 定案的 KV 对齐 + 重复 + GPU 采样） |
| `run_cmp_offline_curve.sh` | 离线吞吐曲线 |
| `run_official.sh` | 跑官方 `bench.py` 口径（133,966 tokens 那个表） |
| `run_nsys_steady.sh` / `run_nsys_nodes.sh` / `run_nsys_reports.sh` | 稳态取样的 nsys 三连 |
| `run_roofline.sh` / `run_onchip_math.sh` / `run_attn_share.sh` / `run_gemm_sweep.sh` | roofline 常数、片上算力、attention 占比、GEMM 扫描 |
| `diag_n8.py` / `diag_vllm_step.py` | 冷启动污染诊断（2627ms → 167ms 那次） |
| `smoke_*.sh` | 冒烟（vLLM 起服务、两种 profiler、离线/在线各一遍） |

### helper / 一次性诊断

`probe_vllm_api.sh`、`introspect_vllm.sh`、`vllm_defaults*.sh`、`vllm_logflag.sh`、
`probe_profiling.sh`、`probe_cmp_env.sh`、`check_official_dataset.py`、`check_official_scripts.sh`、
`read_official.py`、`read_results.py`、`summarize_results.py`、`verify_results.py`、
`run_shortctx_cmp.sh`、`run_final.sh`、`fix_web_refs*.py`（移除前端时的一次性文档改写记录）。

## logs/

| 文件 | 内容 |
|---|---|
| `run_cmp.out` / `run_cmp2.out` / `run_cmp3.out` / `run_cmp4.out` | 对标逐轮原始输出（几万行，含中途失败的轮次） |
| `run_official.out` | 官方 `bench.py` 口径的原始输出 |
| `smoke_vllm2.out` | vLLM 冒烟输出 |

日志是**当时的原始产物**，里面的绝对路径、失败重试、被推翻的中间结论都保持原样，不做美化。
