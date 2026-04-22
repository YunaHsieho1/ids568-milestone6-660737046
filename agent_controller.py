#!/usr/bin/env python3
"""
Part 2 agent — batch trace generator (non-interactive).

Interactive walkthrough: use `agent_controller.ipynb`.
Equivalent CLI: `python scripts/run_agent_traces.py`
"""

from pathlib import Path

from agent_runner import run_all_tasks


def main() -> None:
    root = Path(__file__).resolve().parent
    run_all_tasks(
        root,
        root / "data" / "agent_tasks.json",
        root / "agent_traces",
        model="mistral:7b-instruct",
    )


if __name__ == "__main__":
    main()
