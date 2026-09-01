# -*- coding: utf-8 -*-
"""Generate gRPC Python stubs from nanovllm.proto.

Usage:
    python nanovllm/serving/generate_proto.py
"""
from pathlib import Path
import subprocess
import sys


def main():
    proto_dir = Path(__file__).parent / "proto"
    out_dir = Path(__file__).parent
    proto_file = proto_dir / "nanovllm.proto"

    subprocess.check_call([
        sys.executable, "-m", "grpc_tools.protoc",
        "-I", str(proto_dir),
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ])

    # Make the generated grpc stub importable as part of the nanovllm.serving
    # package instead of requiring the serving directory to be on sys.path.
    grpc_stub = out_dir / "nanovllm_pb2_grpc.py"
    text = grpc_stub.read_text(encoding="utf-8")
    text = text.replace(
        "import nanovllm_pb2 as nanovllm__pb2",
        "from nanovllm.serving import nanovllm_pb2 as nanovllm__pb2",
    )
    grpc_stub.write_text(text, encoding="utf-8")
    print(f"Generated: {out_dir / 'nanovllm_pb2.py'}, {grpc_stub}")


if __name__ == "__main__":
    main()
