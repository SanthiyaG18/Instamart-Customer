"""TF-IDF policy RAG and a checkpointed offline ReAct refund agent."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import sys
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mock_llm

OUT = ROOT / "outputs"
STATE = ROOT / "state"
OUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)

_DEFAULT_INDEX = None


def build_chunks(size=40, overlap=10):
    """Split each policy into overlapping word chunks with source metadata."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be between 0 and size")

    chunks = []
    step = size - overlap
    for path in sorted((ROOT / "data/policies").glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        for start in range(0, len(words), step):
            part = words[start:start + size]
            if part:
                chunks.append({"source": path.name, "text": " ".join(part)})
            if start + size >= len(words):
                break
    return chunks


def build_index(overlap=10):
    chunks = build_chunks(size=40, overlap=overlap)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return chunks, vectorizer, matrix


def _default_index():
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = build_index()
    return _DEFAULT_INDEX


def retrieve(query, k=3, chunks=None, vectorizer=None, matrix=None):
    """Return the k highest-scoring policy chunks for a query."""
    if chunks is None and vectorizer is None and matrix is None:
        chunks, vectorizer, matrix = _default_index()
    elif chunks is None or vectorizer is None or matrix is None:
        raise ValueError("chunks, vectorizer, and matrix must be supplied together")
    if k < 1:
        raise ValueError("k must be at least 1")

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix)[0]
    best_ids = similarities.argsort()[::-1][:k]
    return [
        {
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"],
        }
        for index in best_ids
    ]


def _answer_with_index(question, chunks, vectorizer, matrix, k=3):
    documents = retrieve(question, k, chunks, vectorizer, matrix)
    context = "\n".join(
        f"[{document['source']}]\n{document['text']}" for document in documents
    )
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer only from the supplied context. "
                "If the information is not present, say: "
                "I could not find that information in the policy documents."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
        },
    ]
    response = mock_llm.chat(messages)
    return response.choices[0].message.content or "", documents


def answer(question):
    """Answer from retrieved policy context and return the chunks used."""
    chunks, vectorizer, matrix = _default_index()
    text, documents = _answer_with_index(question, chunks, vectorizer, matrix)
    return {"answer": text, "sources": documents}


def _evaluate_index(qa, chunks, vectorizer, matrix):
    rows = []
    for row in qa.itertuples(index=False):
        answer_text, documents = _answer_with_index(
            row.question, chunks, vectorizer, matrix
        )
        retrieved_sources = [document["source"] for document in documents]
        is_answerable = row.expected_doc != "NONE"
        rows.append(
            {
                "question": row.question,
                "expected_doc": row.expected_doc,
                "expected_fact": row.expected_fact,
                "retrieved_docs": "|".join(retrieved_sources),
                "answer": answer_text,
                "doc_hit": (
                    row.expected_doc in retrieved_sources if is_answerable else False
                ),
                "fact_hit": (
                    bool(row.expected_fact) and row.expected_fact in answer_text
                    if is_answerable
                    else False
                ),
                "fallback_hit": (
                    not is_answerable
                    and answer_text.startswith("I could not find")
                ),
            }
        )
    return pd.DataFrame(rows)


def _print_eval_summary(label, results):
    answerable = results[results.expected_doc.ne("NONE")]
    unanswerable = results[results.expected_doc.eq("NONE")]
    summary = pd.DataFrame(
        [
            {
                "chunking": label,
                "expected_doc_top3": (
                    f"{int(answerable.doc_hit.sum())}/{len(answerable)}"
                ),
                "expected_fact_exact": (
                    f"{int(answerable.fact_hit.sum())}/{len(answerable)}"
                ),
                "unanswerable_fallback": (
                    f"{int(unanswerable.fallback_hit.sum())}/{len(unanswerable)}"
                ),
            }
        ]
    )
    print(summary.to_string(index=False))
    return answerable, unanswerable


def run_rag():
    chunks, vectorizer, matrix = build_index(overlap=10)
    counts = pd.Series([chunk["source"] for chunk in chunks]).value_counts()
    print("Policy chunks:", len(chunks), "by source:", counts.to_dict())
    example = retrieve(
        "refund policy late delivery", chunks=chunks,
        vectorizer=vectorizer, matrix=matrix
    )
    print("Example retrieval:", [(item["source"], round(item["score"], 3)) for item in example])

    qa = pd.read_csv(ROOT / "data/raw/policy_eval.csv", keep_default_na=False)
    results = _evaluate_index(qa, chunks, vectorizer, matrix)
    results.to_csv(OUT / "policy_eval_results.csv", index=False)
    print("\nPolicy evaluation, 40-word chunks with 10-word overlap:")
    _print_eval_summary("overlap 10", results)

    chunks_zero, vectorizer_zero, matrix_zero = build_index(overlap=0)
    results_zero = _evaluate_index(qa, chunks_zero, vectorizer_zero, matrix_zero)
    results_zero.to_csv(OUT / "policy_eval_zero_overlap_results.csv", index=False)
    print("\nPolicy evaluation, 40-word chunks with zero overlap:")
    answerable_zero, _ = _print_eval_summary("overlap 0", results_zero)

    failures = answerable_zero[~answerable_zero.fact_hit]
    if not failures.empty:
        failed = failures.iloc[0]
        print(
            "Zero-overlap boundary example:",
            repr(failed.question),
            "| expected fact:",
            repr(failed.expected_fact),
            "| answer:",
            repr(failed.answer),
        )
        print(
            "The relevant sentence is split across separate 40-word chunks; "
            "retrieval can supply fragments that do not contain the full fact together."
        )
    return results, results_zero


def tool_get_order(order_id):
    with sqlite3.connect(ROOT / "data/instamart.db") as connection:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(
        zip(
            ("order_id", "status", "order_amount", "delivery_minutes"),
            row,
        )
    )


def tool_search_policy(query):
    return {"results": retrieve(query)}


def tool_calculate_refund(order_amount, delivery_minutes):
    refund = (
        round(float(order_amount), 2)
        if delivery_minutes is not None and float(delivery_minutes) > 45
        else 0.0
    )
    return {"refund_inr": refund}


def _write_ledger(tool_name, order_id, refund_inr):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr,
    }
    with (OUT / "refund_ledger.jsonl").open("a", encoding="utf-8") as ledger_file:
        ledger_file.write(json.dumps(record) + "\n")
    return {"refund_inr": refund_inr}


def tool_issue_refund(order_id, refund_inr):
    return _write_ledger("issue_refund", order_id, refund_inr)


def tool_escalate_to_human(order_id, refund_inr):
    return _write_ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order and return its status, amount, and delivery time.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve the three policy chunks most relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate a full refund only for delivery later than 45 minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": ["number", "null"]},
                },
                "required": ["order_amount", "delivery_minutes"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Append an approved simulated refund to the local JSONL ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Append a refund requiring human approval to the local JSONL ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_MAP = {
    "get_order": tool_get_order,
    "search_policy": tool_search_policy,
    "calculate_refund": tool_calculate_refund,
    "issue_refund": tool_issue_refund,
    "escalate_to_human": tool_escalate_to_human,
}


def _checkpoint_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'")
    return STATE / f"{run_id}.json"


def checkpoint(run_id, state, runtime_started=None):
    if runtime_started is not None:
        state["elapsed_seconds"] = state.get("elapsed_before_resume", 0.0) + (
            time.monotonic() - runtime_started
        )
    state["step_number"] = len(state["steps"])
    state["tools_executed"] = [step["tool"] for step in state["steps"]]
    _checkpoint_path(run_id).write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tool=None,
    run_id=None,
    resume=False,
):
    run_id = run_id or scenario_id
    checkpoint_file = _checkpoint_path(run_id)
    if resume:
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"No checkpoint for run {run_id}")
        state = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        if state["scenario_id"] != scenario_id:
            raise ValueError(f"Checkpoint {run_id} belongs to another scenario")
        if state["status"] not in {"SIMULATED_CRASH", "RUNNING"}:
            return state
        state["status"] = "RUNNING"
        state["crash_after_tool"] = None
        state["elapsed_before_resume"] = state.get("elapsed_seconds", 0.0)
    else:
        state = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "user_message": user_message,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "AGENT. Follow policy and use tools. Do not issue refunds "
                        "for cancelled or on-time orders. Refunds above INR 1000 "
                        "require human escalation."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            "steps": [],
            "step_number": 0,
            "tools_executed": [],
            "tokens_used": 0,
            "llm_calls": 0,
            "elapsed_seconds": 0.0,
            "elapsed_before_resume": 0.0,
            "status": "RUNNING",
            "final_answer": "",
            "crash_after_tool": crash_after_tool,
        }

    runtime_started = time.monotonic()
    while state["llm_calls"] < max_iterations:
        elapsed = state.get("elapsed_before_resume", 0.0) + (
            time.monotonic() - runtime_started
        )
        if elapsed >= time_limit_seconds:
            state.update(
                status="FALLBACK_TIMEOUT",
                final_answer="FALLBACK_TIMEOUT: hand off to a human agent.",
            )
            checkpoint(run_id, state, runtime_started)
            return state

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        if state["tokens_used"] > token_limit:
            state.update(
                status="FALLBACK_TOKEN_LIMIT",
                final_answer="FALLBACK_TOKEN_LIMIT: hand off to a human agent.",
            )
            checkpoint(run_id, state, runtime_started)
            return state

        message = response.choices[0].message
        if not message.tool_calls:
            final_answer = message.content or "FALLBACK: no final answer."
            state.update(
                status=final_answer.split(":", 1)[0],
                final_answer=final_answer,
            )
            checkpoint(run_id, state, runtime_started)
            return state

        tool_calls = []
        for call in message.tool_calls:
            arguments = (
                call.arguments
                if isinstance(call.arguments, dict)
                else json.loads(call.arguments)
            )
            tool_calls.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": arguments,
                }
            )
        state["messages"].append(
            {"role": "assistant", "content": None, "tool_calls": tool_calls}
        )

        for call in tool_calls:
            tool_function = TOOL_MAP[call["name"]]
            observation = tool_function(**call["arguments"])
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(observation),
                }
            )
            state["steps"].append(
                {
                    "tool": call["name"],
                    "arguments": call["arguments"],
                    "observation": observation,
                }
            )
            checkpoint(run_id, state, runtime_started)
            if (
                state.get("crash_after_tool") is not None
                and len(state["steps"]) == state["crash_after_tool"]
            ):
                state["status"] = "SIMULATED_CRASH"
                checkpoint(run_id, state, runtime_started)
                return state

    state.update(
        status="FALLBACK_MAX_ITERATIONS",
        final_answer=(
            "FALLBACK_MAX_ITERATIONS: hand off to a human agent "
            "with the recorded steps."
        ),
    )
    checkpoint(run_id, state, runtime_started)
    return state


def resume_agent(run_id):
    checkpoint_file = _checkpoint_path(run_id)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"No checkpoint for run {run_id}")
    saved = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    return run_agent(
        saved["user_message"],
        saved["scenario_id"],
        run_id=run_id,
        resume=True,
    )


def _append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record) + "\n")


def main():
    run_rag()
    scenarios = pd.read_csv(
        ROOT / "data/raw/agent_scenarios.csv", keep_default_na=False
    )
    ledger_path = OUT / "refund_ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()
    agent_runs_path = OUT / "agent_runs.jsonl"
    agent_runs_path.write_text("", encoding="utf-8")

    rows = []
    for row in scenarios.itertuples(index=False):
        result = run_agent(row.message, row.scenario_id)
        rows.append(result)
        _append_jsonl(
            agent_runs_path,
            {
                "scenario_id": row.scenario_id,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "tools": ",".join(step["tool"] for step in result["steps"]),
                "steps": result["steps"],
            },
        )

    first = scenarios.iloc[0]
    token_demo = run_agent(
        first.message, "demo_token_limit", token_limit=200
    )
    timeout_demo = run_agent(
        first.message, "demo_timeout", time_limit_seconds=0
    )
    print("Stop demo token limit:", token_demo["status"])
    print("Stop demo timeout:", timeout_demo["status"])

    crash_demo = run_agent(
        first.message, "demo_crash_resume", crash_after_tool=2
    )
    print("Crash demo first status:", crash_demo["status"])
    resumed = resume_agent("demo_crash_resume")
    print(
        "Crash/resume final:",
        resumed["status"],
        "tools:",
        [step["tool"] for step in resumed["steps"]],
    )
    print(
        pd.DataFrame(
            [
                {
                    "scenario_id": row.scenario_id,
                    "status": result["status"],
                    "refund_inr": next(
                        (
                            step["observation"].get("refund_inr")
                            for step in reversed(result["steps"])
                            if "refund_inr" in step["observation"]
                        ),
                        0.0,
                    ),
                    "llm_calls": result["llm_calls"],
                    "tokens_used": result["tokens_used"],
                }
                for row, result in zip(scenarios.itertuples(index=False), rows)
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
