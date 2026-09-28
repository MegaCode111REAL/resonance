"""Export the exact YourMT3+ multi-channel forward pass for browser inference.

The exported graph is deliberately the teacher-forcing forward pass rather than
Python's autoregressive generate loop. The browser supplies the previously
generated token matrix and performs greedy decoding one token at a time.
This keeps the ONNX graph deterministic and lets ONNX Runtime Web execute the
neural network entirely in the browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


CHECKPOINT = (
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_"
    "b36_nops@last.ckpt"
)


class BrowserForward(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(audio, tokens)["logits"]


def build_model(checkpoint: Path):
    repo_root = Path(__file__).resolve().parents[1]
    yourmt3_root = repo_root / ".build" / "YourMT3"
    source_root = yourmt3_root / "amt" / "src"

    if not yourmt3_root.is_dir():
        raise RuntimeError(
            f"Official YourMT3 source was not found at {yourmt3_root}. "
            "Run the browser model build workflow's source checkout step first."
        )

    if not source_root.is_dir():
        raise RuntimeError(
            f"YourMT3 Python source was not found at {source_root}."
        )

    # model_helper.py lives at the root of the official YourMT3 checkout,
    # while its model/config packages live under amt/src. The official Space
    # adds amt/src to sys.path and imports model_helper from the checkout root.
    sys.path.insert(0, str(yourmt3_root))
    sys.path.insert(0, str(source_root))

    from model_helper import load_model_checkpoint

    args = [
        str(checkpoint),
        "-p", "2024",
        "-tk", "mc13_full_plus_256",
        "-dec", "multi-t5",
        "-nl", "26",
        "-enc", "perceiver-tf",
        "-sqr", "1",
        "-ff", "moe",
        "-wf", "4",
        "-nmoe", "8",
        "-kmoe", "2",
        "-act", "silu",
        "-epe", "rope",
        "-rp", "1",
        "-ac", "spec",
        "-hop", "300",
        "-atc", "1",
        "-pr", "32",
    ]

    model = load_model_checkpoint(args=args, device="cpu")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = build_model(args.checkpoint)
    wrapper = BrowserForward(model).eval()

    # 13 channels are the mc13_full_plus_256 output channels. A single token
    # is enough for tracing; the token axis is exported as dynamic.
    audio = torch.zeros((1, 1, 32767), dtype=torch.float32)
    tokens = torch.zeros((1, 13, 1), dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (audio, tokens),
            args.output,
            input_names=["audio", "tokens"],
            output_names=["logits"],
            dynamic_axes={
                "tokens": {2: "token_length"},
                "logits": {2: "token_length"},
            },
            opset_version=18,
            do_constant_folding=True,
        )

    print(f"Exported {args.output} ({args.output.stat().st_size / 1048576:.1f} MiB)")


if __name__ == "__main__":
    main()
