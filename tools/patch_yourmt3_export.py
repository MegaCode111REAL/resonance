"""Patch official YourMT3 MoE routing so torch.export can trace it."""

from pathlib import Path
import re

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

    pattern = re.compile(
        r"^(?P<indent>[ \t]*)top_x_list = top_x\.tolist\(\)\r?\n"
        r"(?P=indent)idx_list = idx\.tolist\(\)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise SystemExit(
            "Expected YourMT3 MoE Python-list indexing lines were not found"
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}# Keep expert routing indices as tensors. Python .tolist() "
        "creates data-dependent values that torch.export cannot specialize.\n"
        f"{indent}top_x_list = top_x\n"
        f"{indent}idx_list = idx"
    )
    text = pattern.sub(replacement, text, count=1)

    TARGET.write_text(text)
    compile(TARGET.read_text(), str(TARGET), "exec")
    print(f"Patched and syntax-checked {TARGET}")


if __name__ == "__main__":
    main()
