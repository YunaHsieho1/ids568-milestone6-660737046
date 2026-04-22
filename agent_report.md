# Part 2 - Agent Report

**Student:** Tzuyu Hsieh (660737046)  
**Controller:** `agent_runner.py` (LLM-driven tool routing)  
**Model for final runs:** `mistral:7b-instruct` (Ollama, local)  
**Traces:** `agent_traces/t01_trace.json` to `agent_traces/t10_trace.json`

## 1) Tool selection policy

The planner outputs one JSON action per step:
- `retrieve_wiki`: query FAISS corpus from Part 1
- `summarize_text`: compress long evidence
- `finish`: return final answer and source filenames

Decision logic is observable in each trace step through `thought`, `tool`, `arguments`, and tool observations.

## 2) Retrieval integration

`retrieve_wiki` returns structured hits (`source`, `score`, `preview`).  
Those results are either summarized (`summarize_text`) or used directly for `finish`.  
`finish` is expected to cite only filenames seen in retrieval history.

## 3) Performance on 10 tasks

Current batch summary (from the latest `agent_traces/t01_trace.json`–`t10_trace.json` on disk):
- traces written: **10/10**
- episodes whose **last step is `finish`** with a recorded `final_answer`: **9/10**
- **t02** (`Two-step compare BI and business analytics`) hit the step cap (**10** steps) and ends on `summarize_text` (no terminal `finish` in the trace), after several `summarize_text_missing_text` retries / misrouted arguments in mid-episode.
- average steps (all tasks): **4.2** (per-task step counts: t01=3, t02=10, t03=3, t04=4, t05=3, t06=4, t07=2, t08=4, t09=4, t10=5)

Per-task traces are available in `agent_traces/` for decision sequence and evidence trail.

Latency ranges from traces:
- planner: ~2.5 s to ~53.5 s per step
- retrieval: ~91 ms to ~666 ms
- summarization: ~6.4 s to ~19.2 s
- episode wall time: ~14.8 s to ~232.8 s (long tail driven by **t02**)

When running with `HF_HUB_OFFLINE=1`, SentenceTransformers may print a harmless `embeddings.position_ids | UNEXPECTED` load report for `all-MiniLM-L6-v2`; retrieval still completes.

## 4) Failure analysis

Main failure modes observed before controller guards:
- blank or invalid tool names
- empty `finish` payload (`answer` / `sources`)
- empty `summarize_text` input

Mitigation implemented:
- tool-name normalization/inference
- validation for `finish` fields
- validation for summarize text input

These checks improved run completion but retries still occur in some tasks (see **t02** in the traces for repeated invalid `summarize_text` calls before the step budget).

## 5) Model quality/latency tradeoffs

- Strength: with controller constraints, the 7B model can complete multi-step tool workflows.
- Weakness: first-pass JSON/tool-field consistency is not always reliable.
- Tradeoff: local 7B inference is reproducible but slower than GPU-hosted serving.

## Reproduce

```bash
source .venv/bin/activate
# Optional: avoid Hugging Face hub calls if weights are cached locally
HF_HUB_OFFLINE=1 python scripts/run_agent_traces.py
```
