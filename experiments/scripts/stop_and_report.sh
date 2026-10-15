#!/bin/bash
# Stop any leftover benchmark processes and report state.
# Patterns use a bracket so pkill -f does not match this script's own cmdline.
pkill -f 'run_cmp[3].sh' 2>/dev/null
pkill -f 'cmp_online_clien[t]' 2>/dev/null
pkill -f 'cmp_offline_nanovll[m]' 2>/dev/null
pkill -f 'cmp_offline_vll[m]' 2>/dev/null
pkill -f 'cmp_serve_nanovll[m]' 2>/dev/null
sleep 3
pkill -f 'vllm serv[e]' 2>/dev/null
pkill -f 'VLLM::EngineCor[e]' 2>/dev/null
sleep 4
echo "--- remaining procs ---"
ps aux | grep -E 'cmp_|vllm serv' | grep -v grep | head
echo "--- gpu ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- results in ~/nano-vllm-plus/results ---"
ls -1 ~/nano-vllm-plus/results/*.json 2>/dev/null
echo "--- probe_kv ---"
ls -1 ~/nano-vllm-plus/results/probe_kv/*.json 2>/dev/null
