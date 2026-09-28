"""Emit the exact YourMT3 token/event mapping for the browser decoder."""
from __future__ import annotations
import json, sys
from pathlib import Path

def main() -> None:
    root = Path(".build/YourMT3")
    sys.path.insert(0, str(root / "amt" / "src"))
    from config.vocabulary import program_vocab_presets
    from utils.task_manager import TaskManager
    from utils.tokenizer import EventTokenizer
    task = TaskManager(task_name="mc13_full_plus_256")
    tokenizer = EventTokenizer()
    events = []
    for token_id in range(task.num_tokens):
        try:
            decoded = tokenizer.decode([token_id])
            event = decoded[0] if decoded else None
            events.append({
                "id": token_id,
                "type": getattr(event, "type", None),
                "value": int(getattr(event, "value")) if getattr(event, "value", None) is not None else None,
            })
        except Exception:
            events.append({"id": token_id, "type": None, "value": None})
    payload = {
        "model": "YPTF.MoE+Multi (noPS)",
        "task": "mc13_full_plus_256",
        "num_tokens": task.num_tokens,
        "channels": task.num_decoding_channels,
        "events": events,
        "program_vocab": {
            str(program): list(programs)
            for program, programs in program_vocab_presets["mt3_full_plus"].items()
        },
    }
    Path("dist").mkdir(exist_ok=True)
    Path("dist/yourmt3-vocab.json").write_text(
        json.dumps(payload, separators=(",", ":"), default=int), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
