"""Export YourMT3+ as separate browser encoder and decoder ONNX graphs.

The original single forward graph re-ran the complete audio encoder for every
autoregressive token. This exporter splits the exact official YourMT3 forward
path at the encoder/decoder boundary so the browser encodes each audio segment
once and then runs only the decoder for each generated prefix.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch


CHECKPOINT = (
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_"
    "b36_nops@last.ckpt"
)
CHANNELS = 13
AUDIO_SAMPLES = 32767


class BrowserEncoder(torch.nn.Module):
    """Exact YourMT3 audio -> pre-decoder hidden-state path."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.spectrogram = model.spectrogram
        self.pre_encoder = model.pre_encoder
        self.encoder = model.encoder
        self.pre_decoder = model.pre_decoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(audio)
        x = self.pre_encoder(x)
        enc_hs = self.encoder(inputs_embeds=x)["last_hidden_state"]
        return self.pre_decoder(enc_hs)


class BrowserDecoder(torch.nn.Module):
    """Exact YourMT3 teacher-forced decoder + LM head."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.shift_right_fn = model.shift_right_fn
        self.embed_tokens = model.embed_tokens
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.tie_word_embeddings = bool(model.model_cfg["tie_word_embeddings"])
        self.d_model = model.model_cfg["decoder"][model.decoder_type]["d_model"]

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        dec_input_ids = self.shift_right_fn(target_tokens)
        dec_inputs_embeds = self.embed_tokens(dec_input_ids)
        dec_hs, _ = self.decoder(
            inputs_embeds=dec_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        if self.tie_word_embeddings:
            dec_hs = dec_hs * (self.d_model ** -0.5)
        return self.lm_head(dec_hs)


def build_model(checkpoint: Path):
    repo_root = Path(__file__).resolve().parents[1]
    yourmt3_root = repo_root / ".build" / "YourMT3"
    source_root = yourmt3_root / "amt" / "src"

    if not yourmt3_root.is_dir():
        raise RuntimeError(
            f"Official YourMT3 source was not found at {yourmt3_root}."
        )
    if not source_root.is_dir():
        raise RuntimeError(f"YourMT3 Python source was not found at {source_root}.")
    if not checkpoint.is_file():
        raise RuntimeError(f"YourMT3 checkpoint was not found: {checkpoint}")

    checkpoint_dir = (
        yourmt3_root / "amt" / "logs" / "2024" / CHECKPOINT / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    normalized_checkpoint = checkpoint_dir / "last.ckpt"
    if checkpoint.resolve() != normalized_checkpoint.resolve():
        if not normalized_checkpoint.exists():
            shutil.copy2(checkpoint, normalized_checkpoint)

    sys.path.insert(0, str(yourmt3_root))
    sys.path.insert(0, str(source_root))

    from model_helper import load_model_checkpoint

    import os

    os.chdir(yourmt3_root)

    args = [
        CHECKPOINT,
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


def export_model(wrapper, inputs, output: Path, input_names, output_names, dynamic_shapes):
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        program = torch.onnx.export(
            wrapper.eval(),
            inputs,
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            external_data=False,
            optimize=True,
            verify=False,
        )
    program.save(output)
    print(f"Exported {output} ({output.stat().st_size / 1048576:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-output", type=Path, required=True)
    parser.add_argument("--decoder-output", type=Path, required=True)
    args = parser.parse_args()

    model = build_model(args.checkpoint)

    audio = torch.zeros((1, 1, AUDIO_SAMPLES), dtype=torch.float32)
    tokens = torch.zeros((1, CHANNELS, 1), dtype=torch.long)

    encoder = BrowserEncoder(model)
    encoder_output = encoder(audio)
    print("Encoder output shape:", tuple(encoder_output.shape))

    export_model(
        encoder,
        (audio,),
        args.encoder_output,
        ["audio"],
        ["encoder_hidden_states"],
        None,
    )

    decoder = BrowserDecoder(model)
    export_model(
        decoder,
        (encoder_output, tokens),
        args.decoder_output,
        ["encoder_hidden_states", "target_tokens"],
        ["logits"],
        (
            {0: None, 1: None, 2: None, 3: None},
            {0: None, 1: None, 2: "token_length"},
        ),
    )


if __name__ == "__main__":
    main()
