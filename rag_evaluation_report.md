# Part 1 - RAG Evaluation Report

**Course:** IDS 568 Milestone 6  
**Notebook:** `rag_pipeline.ipynb`  
**Corpus:** 7 English Wikipedia files (`data/corpus/wiki_*.txt`)  
**LLM:** `mistral:7b-instruct` with Ollama

## 1) What I implemented

My Part 1 pipeline includes:
- loading documents from `data/corpus/`
- chunking (`chunk_size=450`, `overlap=90`)
- embeddings with `all-MiniLM-L6-v2`
- FAISS index (`IndexFlatIP`, normalized vectors)
- top-k retrieval (`k=4`)
- grounded generation using local Ollama

## 2) Retrieval evaluation (10 queries)

I used `data/eval_queries.json` (each query has `relevant_sources`).

Metric:
- file-level hit@k proxy (count as hit if any top-k source matches expected source)

Result:
- 10/10 hits
- mean hit@k proxy = 1.000

This is a weak metric compared to chunk-level labeling, so it may look optimistic.

## 3) Grounding and failure analysis

Observed behavior:
- If context is missing key info, the model can abstain.
- If retrieval is loosely related, the answer can be plausible but not strongly grounded.

Error attribution:
- Retrieval issues: corpus mismatch / weak retrieval context.
- Generation issues: over-generalization from weak context.

## 4) Latency

Observed ranges from notebook runs:
- embedding all chunks: ~6 s (after warm-up)
- retrieval per query: ~7 ms to ~1.2 s
- generation: ~14 s to ~31 s
- end-to-end: ~14 s to ~32 s

Main bottleneck is generation latency, not retrieval.

## 5) Model deployment details

- model: `mistral:7b-instruct`
- size class: 7B
- serving: Ollama local API
- runtime: macOS, MacBook Air M2, Python 3.12

## 6) Limitations

1. File-level hit@k is a coarse metric.
2. Retrieval quality depends on corpus coverage.
3. Better abstention/confidence control would improve grounding reliability.
