# nano-vllm serving layer

gRPC + Redis + Prometheus serving layer for nano-vllm.

## Environment

Recommended: use the local virtualenv created from the conda `ml` environment:

```powershell
# Create once (already done in this workspace)
C:\Users\Adam\.conda\envs\ml\python.exe -m venv --without-pip --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-serving.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Then run all commands with `.\.venv\Scripts\python.exe`.

If dependencies were installed into `.deps` (sandbox workaround), run commands with:

```bash
export PYTHONPATH="$PWD/.deps:$PWD"   # Linux/macOS
$env:PYTHONPATH="$PWD\.deps;$PWD"     # Windows PowerShell
```

## Generate gRPC stubs

```bash
python nanovllm/serving/generate_proto.py
```

## Start server

GPU-free mock mode (no model, no GPU needed):

```bash
python -m nanovllm.serving --mock --port 50051 --metrics-port 9100
```

Real GPU mode:

```bash
# GPU environment required to actually run the engine.
python -m nanovllm.serving --model /path/to/Qwen3-0.6B --port 50051 --metrics-port 9100
```

### `--max-workers` caps your concurrency

A streaming `Infer` occupies one gRPC worker thread for the **entire** stream, so
`--max-workers` is not a thread-pool tuning knob: it is the hard ceiling on
concurrent requests, which is also the ceiling on the batch the engine can form.

| `--max-workers` | behaviour |
|---|---|
| 10 (old default) | engine never batched past 8–10 sequences; throughput plateaued at ~800 tok/s while the engine alone sustains ~3600 tok/s at batch 32 |
| 64 (default) | 857 → 2930 tok/s at concurrency 64; p99 2366 → 691 ms |
| 256 | reaches 3803 tok/s at concurrency 128; concurrency 256 is refused by the memory backpressure instead of OOMing (KV pool tops out at ~162 concurrent sequences) |

Keep it comfortably above the concurrency you intend to serve.

Optional Redis multi-instance registration:

```bash
python -m nanovllm.serving \
  --model /path/to/Qwen3-0.6B \
  --port 50051 \
  --instance-id instance-1 \
  --redis-url redis://localhost:6379/0 \
  --addr 10.0.0.1:50051
```

## Routing gateway (multi-instance)

Start a local Redis with:

```bash
docker compose up -d redis
```

After engine instances register themselves in Redis, start the gateway:

```bash
python -m nanovllm.serving --gateway --redis-url redis://localhost:6379/0 --port 50050
```

Clients should talk to the gateway (`localhost:50050`) instead of engine instances.

## Test client

```bash
python -m nanovllm.serving.client --target localhost:50051 --prompt "Hello"
```

## Benchmark

```bash
python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 64 --concurrency 8
```
