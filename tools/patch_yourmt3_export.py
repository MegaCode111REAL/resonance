"""Patch official YourMT3 MoE routing so torch.export can trace it."""

from pathlib import Path

TARGET = Path(".build/YourMT3/amt/src/model/ff_layer.py")


def main() -> None:
    text = TARGET.read_text()

    old_guard = "if top_x.shape[0] == 0:"
    new_guard = (
        "if torch.fx.experimental.symbolic_shapes."
        "guard_size_oblivious(top_x.shape[0] == 0):"
    )
    if old_guard not in text:
        raise SystemExit("Expected YourMT3 MoE empty-expert guard was not found")
    text = text.replace(old_guard, new_guard, 1)

    old_top_x = "top_x_list = top_x.tolist()"
    old_idx = "idx_list = idx.tolist()"
    if old_top_x not in text or old_idx not in text:
        raise SystemExit(
            "Expected YourMT3 MoE Python-list indexing lines were not found"
        )

    text = text.replace(
        old_top_x,
        "# Keep expert routing indices as tensors. Python .tolist() creates\n"
        "# data-dependent values that torch.export cannot specialize.\n"
        "top_x_list = top_x",
        1,
    )
    text = text.replace(old_idx, "idx_list = idx", 1)

    TARGET.write_text(text)
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
