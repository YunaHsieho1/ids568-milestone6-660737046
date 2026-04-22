"""
Part 2 agent: LLM chooses between retrieve_wiki and summarize_text, then may finish with final_answer.
Requires Ollama (same model as Part 1) for routing + summarization + optional synthesis.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from rag_core import RAGPipeline

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "mistral:7b-instruct"


def ollama_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    *,
    num_predict: int = 2048,
) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        timeout=600,
    )
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json().get("response", "").strip(), dt


def _unwrap_markdown_json_fence(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def extract_first_json_value(s: str, start: int) -> str:
    """First balanced {...} or [...] slice starting at start (string-aware; handles nested arrays)."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    raise ValueError("unclosed JSON value")


def parse_agent_json(text: str) -> dict[str, Any]:
    """Extract the first complete top-level JSON object from model output (ignores trailing prose / extra JSON)."""
    s = _unwrap_markdown_json_fence(text.strip())
    brace = s.find("{")
    if brace < 0:
        raise json.JSONDecodeError("No JSON object found in model output", s, 0)
    fragment = extract_first_json_value(s, brace)
    return json.loads(fragment)


def coerce_retrieve_query(raw: Any) -> str:
    """Planner sometimes emits a dict for query; flatten to a single search string."""
    if isinstance(raw, dict):
        parts: list[str] = []
        for k, v in raw.items():
            if v is None:
                continue
            parts.append(f"{k} {v}" if str(k).isidentifier() else f"{v}")
        return " ".join(parts) if parts else json.dumps(raw, ensure_ascii=False)
    return str(raw)


_CANON_TOOLS = frozenset({"retrieve_wiki", "summarize_text", "finish"})


def normalize_planner_tool(tool: str, args: dict[str, Any]) -> str:
    """Map aliases and infer tool when the model leaves `tool` blank but shapes `arguments`."""
    raw = (tool or "").strip()
    t = raw.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "retrievewiki": "retrieve_wiki",
        "retrieve": "retrieve_wiki",
        "search": "retrieve_wiki",
        "search_wiki": "retrieve_wiki",
        "wiki_retrieve": "retrieve_wiki",
        "wiki_search": "retrieve_wiki",
        "summarizetext": "summarize_text",
        "summarize": "summarize_text",
        "summary": "summarize_text",
        "done": "finish",
        "complete": "finish",
    }
    t = aliases.get(t, t)
    if t in _CANON_TOOLS:
        return t
    # Infer from argument keys (common failure: tool "" with only query/k).
    if "answer" in args or "sources" in args:
        return "finish"
    if "query" in args:
        return "retrieve_wiki"
    if "text" in args:
        return "summarize_text"
    return raw


def normalize_finish_arguments(args: dict[str, Any]) -> dict[str, Any]:
    out = dict(args)
    ans = out.get("answer")
    if ans is not None and not isinstance(ans, str):
        out["answer"] = str(ans).strip()
    elif isinstance(ans, str):
        out["answer"] = ans.strip()
    src = out.get("sources")
    if isinstance(src, str) and src.strip():
        out["sources"] = [src.strip()]
    elif isinstance(src, list):
        out["sources"] = [str(x).strip() for x in src if str(x).strip()]
    elif src is not None:
        s = str(src).strip()
        out["sources"] = [s] if s else []
    else:
        out["sources"] = []
    return out


def finish_arguments_ok(args: dict[str, Any]) -> bool:
    ans = args.get("answer", "")
    src = args.get("sources", [])
    if not isinstance(ans, str) or not ans.strip():
        return False
    if not isinstance(src, list) or not src:
        return False
    return all(isinstance(x, str) and x.strip() for x in src)


def build_planner_prompt(mission: str, history: list[dict]) -> str:
    hist = json.dumps(history, ensure_ascii=False, indent=2)[:14000]
    return f"""You are the controller for a two-tool MSBA research assistant.

Available tools (pick exactly one tool name per step):
- "retrieve_wiki" — arguments: {{"query": "<single English search string>", "k": <int 2-6>}}. Corpus files are named like wiki_*.txt.
- "summarize_text" — arguments: {{"text": "<plain text to compress>"}}. Use to shorten long passages (e.g., after retrieval) into a brief bullet summary.
- "finish" — arguments: {{"answer": "<final answer in English>", "sources": ["<filename>", ...]}}. Use when you can answer the user mission from evidence already in the conversation history.

Rules:
- Prefer retrieve_wiki first if the mission needs facts from the corpus.
- Use summarize_text when the user explicitly wants a short summary, or when retrieved passages are very long and you need a compact version before finishing.
- Never invent citations: sources must be filenames you actually saw in retrieve results in history (must match wiki_*.txt names).
- For **finish**, you MUST include non-empty **arguments.answer** (English) and **arguments.sources** (a JSON array of wiki_*.txt filenames from prior retrieve hits).
- **This step only:** choose the **single next** action. Do NOT describe future steps, do NOT output multiple JSON objects, do NOT wrap output in chat roles or arrays.
- **Output format (critical):** reply with **one** JSON object and **nothing else** — no markdown fences, no "(Assuming ...)" commentary, no prose before or after the JSON.
- For summarize_text, arguments.text must be **valid JSON string** content: a **short** plain-English paragraph you type yourself (under ~800 characters), paraphrasing or quoting from History previews — **never** Python list syntax, **never** `.join`, **never** paste multi-line code inside the string.
- If History contains a **controller** message, treat it as a hard constraint and fix the issue before choosing the next tool.

User mission:
{mission}

History (previous steps; observations may truncate long text):
{hist}
""".strip()


def run_episode(
    mission: str,
    pipeline: RAGPipeline,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = 10,
    top_k_default: int = 4,
) -> dict[str, Any]:
    history: list[dict] = []
    trace_steps: list[dict] = []
    t_episode = time.perf_counter()

    for step in range(1, max_steps + 1):
        prompt = build_planner_prompt(mission, history)
        raw, t_plan = ollama_chat(prompt, model=model, temperature=0.05, num_predict=3072)
        try:
            decision = parse_agent_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            repair = (
                "Your previous assistant message was not valid JSON for this API.\n"
                "Return ONLY one JSON object with keys: \"thought\", \"tool\", \"arguments\".\n"
                "No markdown code fences. No text before or after the JSON. No second JSON object.\n"
                "For summarize_text, arguments.text must be one short plain-English paragraph (paraphrase History); "
                "no Python lists, no .join, no unescaped line breaks inside the JSON string.\n\n"
                "Broken message (fix syntax; keep the same next tool intent):\n"
            )
            repair_prompt = repair + raw[:2200]
            raw_retry, t_retry = ollama_chat(
                repair_prompt, model=model, temperature=0.0, num_predict=1536
            )
            t_plan += t_retry
            try:
                decision = parse_agent_json(raw_retry)
                raw = raw_retry
            except (json.JSONDecodeError, ValueError) as e2:
                trace_steps.append(
                    {
                        "step": step,
                        "error": "planner_json_parse_failed",
                        "raw": raw[:2000],
                        "raw_retry": raw_retry[:2000],
                        "exception": f"{e!s}; retry: {e2!s}",
                    }
                )
                break

        thought = decision.get("thought", "")
        tool_raw = decision.get("tool")
        if tool_raw is None:
            tool_raw = ""
        args = decision.get("arguments")
        if not isinstance(args, dict):
            args = {}
        tool = normalize_planner_tool(str(tool_raw), args)
        if tool == "finish":
            args = normalize_finish_arguments(args)

        step_record: dict[str, Any] = {
            "step": step,
            "planner_ms": t_plan * 1000,
            "thought": thought,
            "tool": tool,
            "arguments": args,
        }
        tr = str(tool_raw).strip()
        if tr and tr.replace(" ", "_").lower() != tool:
            step_record["tool_raw"] = tool_raw
        if not tr and tool in _CANON_TOOLS:
            step_record["tool_inferred"] = True
        trace_steps.append(step_record)

        if tool == "finish":
            if not finish_arguments_ok(args):
                step_record["error"] = "finish_invalid_empty_fields"
                history.append(
                    {
                        "role": "controller",
                        "message": (
                            "Your last JSON used tool finish but arguments.answer was empty or arguments.sources "
                            "was missing/empty. Reply again with ONE JSON object: tool must be finish, and "
                            "arguments must include a non-empty English answer plus sources as a non-empty JSON array "
                            "of wiki_*.txt filenames that appear in prior retrieve_wiki hit lists in History."
                        ),
                    }
                )
                continue
            trace_steps[-1]["final_answer"] = args.get("answer", "")
            trace_steps[-1]["claimed_sources"] = args.get("sources", [])
            break

        if tool == "retrieve_wiki":
            q = coerce_retrieve_query(args.get("query", mission))
            k = int(args.get("k", top_k_default))
            k = max(2, min(8, k))
            hits, t_ret = pipeline.retrieve(q, k)
            obs = {
                "query": q,
                "k": k,
                "retrieval_ms": t_ret * 1000,
                "hits": [
                    {
                        "chunk_id": h["chunk_id"],
                        "source": h["source"],
                        "score": h["score"],
                        "preview": h["text"][:600],
                    }
                    for h in hits
                ],
            }
            history.append({"role": "tool", "tool": "retrieve_wiki", "observation": obs})
            trace_steps[-1]["observation"] = obs
            continue

        if tool == "summarize_text":
            text = str(args.get("text", "")).strip()[:12000]
            if not text:
                trace_steps[-1]["error"] = "summarize_text_missing_text"
                history.append(
                    {
                        "role": "controller",
                        "message": (
                            "Your last JSON used summarize_text but arguments.text was empty. Either call "
                            "retrieve_wiki first (if you need evidence), or provide a short plain-English paragraph "
                            "in arguments.text copied/paraphrased from History retrieve previews — never call "
                            "summarize_text with an empty string."
                        ),
                    }
                )
                continue
            sprompt = (
                "Summarize the following text in at most 5 bullet points. "
                "Use English. Do not add facts not present in the text.\n\n" + text
            )
            summary, t_sum = ollama_chat(sprompt, model=model, temperature=0.1, num_predict=1536)
            obs = {"summarize_ms": t_sum * 1000, "summary": summary}
            history.append({"role": "tool", "tool": "summarize_text", "observation": obs})
            trace_steps[-1]["observation"] = obs
            continue

        trace_steps[-1]["error"] = f"unknown_tool:{tr or repr(tool_raw)}"
        break

    return {
        "mission": mission,
        "model": model,
        "elapsed_s": time.perf_counter() - t_episode,
        "steps": trace_steps,
    }


def load_tasks(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if not raw.strip():
        raise ValueError(
            f"Tasks file is empty: {path.resolve()}. "
            "Restore data/agent_tasks.json (or re-save from repo) and retry."
        )
    text = raw.decode("utf-8-sig").strip()
    data = json.loads(text)
    return list(data)


def run_all_tasks(
    root: Path,
    tasks_path: Path,
    out_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
) -> None:
    pipeline = RAGPipeline(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tasks_path)
    for t in tasks:
        tid = t["id"]
        mission = t["mission"]
        trace = run_episode(mission, pipeline, model=model)
        trace["task_id"] = tid
        trace["task_title"] = t.get("title", "")
        out = out_dir / f"{tid}_trace.json"
        out.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote", out, "steps=", len(trace["steps"]))
