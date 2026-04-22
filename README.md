# IDS568 Milestone 6

This repo contains my Part 1 (RAG) and Part 2 (agent) implementation.

## Files to submit (root level)

- `README.md`
- `rag_pipeline.ipynb`
- `rag_evaluation_report.md`
- `rag_pipeline_diagram.md`
- `agent_controller.py`
- `agent_report.md`
- `agent_traces/`
- `requirements.txt`

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
pip install -r requirements.txt
```

## Model used for final runs

- Model: `mistral:7b-instruct`
- Size class: 7B
- Serving: Ollama (`http://127.0.0.1:11434/api/generate`)
- Runtime: macOS, MacBook Air M2 (16 GB), Python 3.12
- Typical generation latency: about 16-32 seconds per short answer

```bash
ollama pull mistral:7b-instruct
```

## Part 1

Run:

```bash
jupyter notebook rag_pipeline.ipynb
```

Includes:
- ingestion from `data/corpus/`
- chunking
- embeddings (`all-MiniLM-L6-v2`)
- FAISS index
- retrieval + grounded generation

Part 1 report and diagram:
- `rag_evaluation_report.md`
- `rag_pipeline_diagram.md`

## Part 2

Run:

```bash
python scripts/run_agent_traces.py
```

Equivalent entrypoint:

```bash
python agent_controller.py
```

To regenerate a single trace (for example after a corrupted or cloud-placeholder file):

```bash
python scripts/run_one_trace.py t10
```

Outputs:
- `agent_traces/t01_trace.json` to `agent_traces/t10_trace.json`
- summary in `agent_report.md`

Example usage:
- Part 1: run `rag_pipeline.ipynb`
- Part 2: run `python scripts/run_agent_traces.py` to generate all agent traces

## Architecture overview

- `rag_core.py`: shared corpus loading, chunking, embedding, FAISS indexing, retrieval
- `agent_runner.py`: planner loop with `retrieve_wiki`, `summarize_text`, `finish`
- `agent_controller.py`: entry point for running all tasks

## Known limitations

- Planner can still require retries for strict JSON/tool schema.
- Retrieval quality depends on corpus coverage and query wording.
- Local CPU inference has noticeable latency for generation.
