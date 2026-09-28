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

    old_indexing = (
        "top_x_list = top_x.tolist()\n"
        "idx_list = idx.tolist()"
    )
    new_indexing = (
        "# Keep expert routing indices as tensors. Python .tolist() creates\n"
        "# data-dependent values that torch.export cannot specialize.\n"
        "top_x_list = top_x\n"
        "idx_list = idx"
    )
    if old_indexing not in text:
        raise SystemExit(
            "Expected YourMT3 MoE Python-list indexing block was not found"
        )
    text = text.replace(old_indexing, new_indexing, 1)

    TARGET.write_text(text)
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
