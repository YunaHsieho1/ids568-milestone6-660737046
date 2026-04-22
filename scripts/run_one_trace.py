#!/usr/bin/env python3
"""Run one Part 2 task by id (e.g. t10) and write agent_traces/<id>_trace.json. Repo root, Ollama up."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runner import load_tasks, run_episode
from rag_core import RAGPipeline


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_one_trace.py <task_id>", file=sys.stderr)
        raise SystemExit(2)
    tid = sys.argv[1].strip()
    tasks_path = ROOT / "data" / "agent_tasks.json"
    tasks = load_tasks(tasks_path)
    selected = [t for t in tasks if t["id"] == tid]
    if not selected:
        known = [t["id"] for t in tasks]
        print(f"Unknown task id {tid!r}. Known: {known}", file=sys.stderr)
        raise SystemExit(1)
    t = selected[0]
    out_dir = ROOT / "agent_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = RAGPipeline(ROOT)
    trace = run_episode(t["mission"], pipeline, model="mistral:7b-instruct")
    trace["task_id"] = tid
    trace["task_title"] = t.get("title", "")
    out = out_dir / f"{tid}_trace.json"
    out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out, "steps=", len(trace["steps"]))


if __name__ == "__main__":
    main()
