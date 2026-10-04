"""Small batch-one prefill using one of the three retained backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import torch
from mbsa import reference_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("reference", "triton", "cutedsl"), default="reference"
    )
    parser.add_argument("--q", type=int)
    parser.add_argument("--kv", type=int)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare accelerator output with the Python oracle",
    )
    args = parser.parse_args()
    qn = args.q if args.q is not None else (32 if args.backend == "reference" else 128)
    kn = (
        args.kv
        if args.kv is not None
        else (160 if args.backend == "reference" else 4096)
    )
    if not 0 < qn <= min(kn, 8192) or kn > 1048576:
        parser.error("requires 0 < Q <= 8192 and Q <= KV <= 1M")
    if args.backend == "triton" and (kn - qn) % 128:
        parser.error("Triton requires a 128-token-aligned KV-Q offset")
    device = "cpu" if args.backend == "reference" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("the accelerator backend requires CUDA")
    generator = torch.Generator(device=device).manual_seed(17)

    def rand(shape):
        return torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        )

    operands = (
        rand((qn, 4, 128)),
        rand((kn, 1, 128)),
        rand((qn, 32, 128)),
        rand((kn, 4, 128)),
        rand((kn, 4, 128)),
    )
    if args.backend == "reference":
        actual = reference_pipeline(*operands)
    elif args.backend == "triton":
        from mbsa import triton_pipeline

        actual = triton_pipeline(*operands)
    else:
        from mbsa import prepare_cutedsl_pipeline

        actual = prepare_cutedsl_pipeline(*operands).run()
    if args.check and args.backend != "reference":
        expected = reference_pipeline(*operands)
        torch.testing.assert_close(actual[0], expected[0], rtol=0.03, atol=0.03)
        torch.testing.assert_close(actual[1], expected[1], rtol=0.002, atol=0.01)
        torch.testing.assert_close(
            actual[2].sort(-1).values, expected[2].sort(-1).values, rtol=0, atol=0
        )
    print(
        json.dumps(
            {
                "backend": args.backend,
                "device": device,
                "output": list(actual[0].shape),
                "lse": list(actual[1].shape),
                "remote_route": list(actual[2].shape),
                "precision_checked": args.check and args.backend != "reference",
            }
        )
    )


if __name__ == "__main__":
    main()
