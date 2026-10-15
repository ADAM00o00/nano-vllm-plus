# -*- coding: utf-8 -*-
"""Group the archived experiment scripts into the three workstreams.

ASCII-only source on purpose (the group names are ASCII); it prints a summary and
fails if any script is unclassified.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "scripts"

GROUPS = {
    "shared": [
        "sync_to_wsl.sh", "pip_fix.py", "archive_paths.py", "group_experiments.py",
        "wsl_probe.sh", "wsl_probe2.sh", "check_nvidia_repo.sh", "fix_nsight_install.sh",
        "probe_flash_attn_constraint.sh", "probe_fa2.sh", "probe_profiling.sh",
        "probe_cmp_env.sh", "vllm_defaults.sh", "vllm_defaults2.sh", "vllm_logflag.sh",
        "introspect_vllm.sh", "probe_vllm_api.sh",
    ],
    "sm89": [
        "run_decode_tests.sh", "run_decode_bench.sh", "run_sweeps.sh", "run_stages.sh",
        "run_splits.sh", "run_kernel_families.sh", "run_attn_share.sh", "run_gemm_sweep.sh",
        "run_roofline.sh", "run_onchip_math.sh", "run_shortctx_cmp.sh",
        "run_ncu_verify.sh", "run_e2e.sh", "run_final.sh",
    ],
    "serving": [
        "run_maxworkers_test.sh", "run_instrumented_server.sh", "run_step_profile.sh",
        "run_final_profile.sh", "run_confirm_fix.sh", "run_verify_fix.sh", "run_boundary.sh",
        "run_online_bench.sh", "run_concurrency_scale.sh", "run_busy_analysis.sh",
        "run_busy_fraction.sh", "run_busy_oneshot.sh", "run_p1.sh", "run_p1b.sh",
        "stop_and_report.sh",
    ],
    "vllm": [
        "probe_kv_size.py", "probe_kv_max.sh", "probe_vllm_kv_max.sh", "probe_vllm_kv_push.sh",
        "verify_kv_pressure.sh", "run_cmp.sh", "run_cmp2.sh", "run_cmp3.sh", "run_cmp4.sh",
        "run_cmp_kernels.sh", "run_cmp_offline_curve.sh", "run_official.sh",
        "check_official_dataset.py", "check_official_scripts.sh", "read_official.py",
        "diag_n8.py", "diag_vllm_step.py", "run_nsys_steady.sh", "run_nsys_nodes.sh",
        "run_nsys_reports.sh", "smoke_cmp_offline.sh",
        "smoke_cmp_online.sh", "smoke_profilers.sh", "smoke_vllm.sh", "smoke_vllm2.sh",
        "read_results.py", "summarize_results.py", "verify_results.py",
    ],
}

all_names = [n for names in GROUPS.values() for n in names]
dupes = sorted({n for n in all_names if all_names.count(n) > 1})
if dupes:
    print("DUPLICATE CLASSIFICATION: " + ", ".join(dupes))
    sys.exit(1)

# scripts kept at experiments/scripts/ root: none yet
mapping = {}
for group, names in GROUPS.items():
    for n in names:
        mapping[n] = group

present = sorted(p.name for p in SRC.glob("*") if p.is_file())
unknown = [n for n in present if n not in mapping]
missing = [n for n in mapping if n not in present]

print(f"present={len(present)} classified={len(present) - len(unknown)}")
if unknown:
    print("UNCLASSIFIED: " + ", ".join(unknown))
    sys.exit(1)
if missing:
    print("listed but absent (ignored): " + ", ".join(missing))

for name, group in sorted(mapping.items()):
    if name not in present:
        continue
    dest = ROOT / "experiments" / group
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(SRC / name), str(dest / name))

print("moved OK")
for group in GROUPS:
    d = ROOT / "experiments" / group
    print(f"  {group:8s} {len([p for p in d.glob('*') if p.is_file()])} files")
