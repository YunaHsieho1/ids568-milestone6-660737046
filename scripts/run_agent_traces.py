#!/usr/bin/env python3
"""Run all Part 2 agent tasks and write JSON traces to agent_traces/. Run from repo root with Ollama up."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runner import run_all_tasks


def main() -> None:
    run_all_tasks(
        ROOT,
        ROOT / "data" / "agent_tasks.json",
        ROOT / "agent_traces",
        model="mistral:7b-instruct",
    )


if __name__ == "__main__":
    main()
