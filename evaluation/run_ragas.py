"""RAG 평가 (README 5.3절).

    python -m evaluation.run_ragas               # Hit@k, MRR (LLM 없음, 유사도 검색)
    python -m evaluation.run_ragas --mmr         # MMR 비교
    python -m evaluation.run_ragas -k 4 --name chunk_section
    python -m evaluation.run_ragas --ragas       # Hit@k + RAGAS (Bedrock 호출)

RAGAS
- Context Precision / Recall: 정답 회의록이 있는 질의(16건)에서 상위 k개 청크와 정답 문장(reference)을 비교
- Faithfulness: Context Agent가 실제로 쓰는 프롬프트와 검색 개수(SEARCH_K)로 결정 충돌 경고를 만들고,
  그 경고가 검색된 청크에 근거하는지 본다. 대상은 결정 충돌 질의(4건)와 관련 회의록 없음 질의(2건)
- 오경고율: 관련 회의록 없음 질의에서 충돌 경고를 만든 비율 (목표 0)

먼저 python -m src.retriever --ingest 로 인덱스를 만들어야 한다.
"""

import argparse
import asyncio
import json
import math
import warnings
from datetime import datetime
from pathlib import Path

from src.retriever import DEFAULT_K, search_past_meetings

DATASET = Path("evaluation/dataset/retrieval.jsonl")
RESULTS_DIR = Path("evaluation/results")

TARGET_HIT = 0.90
TARGET_MRR = 0.75
TARGET_RAGAS = {"context_precision": 0.80, "context_recall": 0.85, "faithfulness": 0.90}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_hit(rows: list[dict], k: int, mmr: bool) -> dict:
    per_query = []
    for row in rows:
        docs = search_past_meetings(row["query"], k=k, before=row.get("before"), mmr=mmr)
        sources = [d.metadata["source"] for d in docs]
        scores = [d.metadata.get("score") for d in docs]
        refs = set(row["reference_sources"])

        rank = next((i for i, s in enumerate(sources, start=1) if s in refs), None)
        per_query.append({
            "id": row["id"],
            "type": row["type"],
            "query": row["query"],
            "hit": rank is not None if refs else None,
            "rank": rank,
            "top_score": scores[0] if scores else None,
            "retrieved": [d.metadata["chunk_id"] for d in docs],
        })

    scored = [q for q in per_query if q["hit"] is not None]
    hit = sum(q["hit"] for q in scored) / len(scored)
    mrr = sum(1 / q["rank"] for q in scored if q["rank"]) / len(scored)

    # "관련 회의록 없음" 판정 기준을 정하기 위해 관련 질의와 무관 질의의 최고 관련도를 비교
    rel_scores = [q["top_score"] for q in scored if q["top_score"] is not None]
    none_scores = [q["top_score"] for q in per_query if q["hit"] is None and q["top_score"] is not None]

    by_type = {}
    for q in scored:
        by_type.setdefault(q["type"], []).append(q["hit"])

    return {
        "k": k,
        "mmr": mmr,
        f"hit@{k}": round(hit, 3),
        "mrr": round(mrr, 3),
        "hit_by_type": {t: f"{sum(v)}/{len(v)}" for t, v in by_type.items()},
        "min_top_score_relevant": min(rel_scores) if rel_scores else None,
        "max_top_score_no_relevant": max(none_scores) if none_scores else None,
        "queries": per_query,
    }


# ── RAGAS ────────────────────────────────────────────────


async def conflict_warnings(row: dict) -> tuple[list[str], list[str]]:
    """Context Agent와 같은 프롬프트·검색 개수로 질의를 '이번 결정'으로 보고 충돌 경고를 만든다.
    (경고 목록, 판단에 쓴 청크 본문)"""
    from langchain_core.prompts import ChatPromptTemplate

    from src.llm import get_llm
    from src.nodes.context import HUMAN, SEARCH_K, SYSTEM, ContextJudgement, _chunk

    # search_past_meetings 도구와 같은 형태로 변환
    hits = [{"source": d.metadata["source"], "date": d.metadata["date"], "section": d.metadata["section"],
             "content": d.page_content}
            for d in search_past_meetings(row["query"], k=SEARCH_K, before=row.get("before"))]
    decisions = f"[D1] {row['query']} (안건: {row['query']})\n" + "\n".join(_chunk(r) for r in hits)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    judged: ContextJudgement = await (prompt | get_llm().with_structured_output(ContextJudgement)).ainvoke(
        {"meeting_date": row.get("before") or "알 수 없음", "decisions": decisions, "items": "(후보 없음)"}
    )
    by_source = {r["source"]: r for r in hits}
    # Context Agent와 같이 검색 결과에 없는 출처는 버린다
    found = [f"{by_source[c.source]['date']} {c.source}: {c.past_decision}"
             for c in judged.conflicts if c.source in by_source]
    return found, [as_context(r["source"], r["content"]) for r in hits]


def ragas_question(query: str) -> str:
    """RAGAS는 user_input을 '질문'으로 보고 청크가 답에 쓸모 있는지 판정한다. 평가셋 질의는 새 결정·할 일
    문장이라 그대로 넘기면 '12개로 늘리는 정보가 없다'며 과거 결정 청크를 무관하다고 본다.
    검색의 실제 목적(과거 맥락 찾기)을 질문으로 적어 넘긴다."""
    return f"과거 회의록에서 다음 내용과 관련된 과거 결정, 기존 할 일, 일정 근거는 무엇인가? 내용: {query}"


def as_context(source: str, content: str) -> str:
    """Agent가 보는 형태 그대로 '(파일명) 본문'. 경고 문구의 출처 파일명을 근거에서 확인할 수 있게 한다."""
    return f"({source}) {content}"


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 3) if vals else None


def _scores(result, ids: list[str]) -> tuple[dict, list[dict]]:
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "retrieved_contexts", "response", "reference")]
    per = [{"id": i, **{c: (None if math.isnan(row[c]) else round(float(row[c]), 3)) for c in metric_cols}}
           for i, (_, row) in zip(ids, df.iterrows())]
    return {c: _mean(df[c].tolist()) for c in metric_cols}, per


def evaluate_ragas(rows: list[dict], k: int) -> dict:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

    from src.llm import get_embeddings, get_llm

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    llm = LangchainLLMWrapper(get_llm())
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    run_config = RunConfig(max_workers=4, timeout=180)

    # 1) Context Precision / Recall: 정답 회의록이 있는 질의
    ctx_rows = [r for r in rows if r["reference_sources"]]
    ctx_samples = [{
        "user_input": ragas_question(r["query"]),
        "retrieved_contexts": [as_context(d.metadata["source"], d.page_content)
                               for d in search_past_meetings(r["query"], k=k, before=r.get("before"))],
        "reference": r["reference"],
    } for r in ctx_rows]
    ctx_result = evaluate(EvaluationDataset.from_list(ctx_samples),
                          metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
                          llm=llm, embeddings=emb, run_config=run_config, show_progress=False)
    ctx_means, ctx_per = _scores(ctx_result, [r["id"] for r in ctx_rows])

    # 2) Faithfulness: Context Agent의 결정 충돌 경고
    warn_rows = [r for r in rows if r["type"] in ("decision_conflict", "no_relevant")]
    outputs = asyncio.run(_gather(conflict_warnings(r) for r in warn_rows))
    warning_log = [{"id": r["id"], "type": r["type"], "query": r["query"], "warnings": w}
                   for r, (w, _) in zip(warn_rows, outputs)]
    faith_items = [(r, w, ctx) for r, (w, ctx) in zip(warn_rows, outputs) if w]
    faith_means, faith_per = {"faithfulness": None}, []
    if faith_items:
        faith_result = evaluate(
            EvaluationDataset.from_list([{"user_input": ragas_question(r["query"]), "retrieved_contexts": ctx,
                                          "response": "\n".join(w)} for r, w, ctx in faith_items]),
            metrics=[Faithfulness()], llm=llm, embeddings=emb, run_config=run_config, show_progress=False)
        faith_means, faith_per = _scores(faith_result, [r["id"] for r, _, _ in faith_items])

    conflict_rows = [w for w in warning_log if w["type"] == "decision_conflict"]
    none_rows = [w for w in warning_log if w["type"] == "no_relevant"]
    return {
        "k": k,
        "context_precision": ctx_means.get("llm_context_precision_with_reference"),
        "context_recall": ctx_means.get("context_recall"),
        "faithfulness": faith_means.get("faithfulness"),
        "conflict_detected": f"{sum(bool(w['warnings']) for w in conflict_rows)}/{len(conflict_rows)}",
        "false_warning_rate": round(sum(bool(w["warnings"]) for w in none_rows) / len(none_rows), 3)
        if none_rows else None,
        "context_per_query": ctx_per,
        "faithfulness_per_query": faith_per,
        "warnings": warning_log,
    }


async def _gather(coros):
    return await asyncio.gather(*coros)


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="검색 평가 (Hit@k, MRR, RAGAS)")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--mmr", action="store_true")
    parser.add_argument("--ragas", action="store_true", help="RAGAS 지표도 측정 (Bedrock 호출)")
    parser.add_argument("--name", default=None, help="결과 파일에 붙일 설정 이름")
    args = parser.parse_args()

    result = evaluate_hit(load_jsonl(DATASET), k=args.k, mmr=args.mmr)

    k = args.k
    print(f"Hit@{k} {result[f'hit@{k}']} (목표 {TARGET_HIT})  MRR {result['mrr']} (목표 {TARGET_MRR})")
    print(f"유형별: {result['hit_by_type']}")
    if not args.mmr:
        print(
            f"최고 관련도 - 관련 질의 최솟값 {result['min_top_score_relevant']}, "
            f"무관 질의 최댓값 {result['max_top_score_no_relevant']}"
        )
    for q in result["queries"]:
        mark = {True: "O", False: "X", None: "-"}[q["hit"]]
        print(f"  {mark} {q['id']} rank={q['rank']} top={q['top_score']}  {q['query']}  → {q['retrieved']}")

    prefix = "hitk"
    if args.ragas:
        prefix = "ragas"
        ragas = evaluate_ragas(load_jsonl(DATASET), k=args.k)
        result["ragas"] = ragas
        print("\nRAGAS")
        for name, target in TARGET_RAGAS.items():
            print(f"  {name:<18} {ragas[name]} (목표 {target})")
        print(f"  결정 충돌 경고     {ragas['conflict_detected']}")
        print(f"  오경고율           {ragas['false_warning_rate']} (목표 0)")
        for w in ragas["warnings"]:
            print(f"    {w['id']} {w['type']:<17} 경고 {len(w['warnings'])}건  {w['warnings'][:1]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or ("mmr" if args.mmr else "similarity")
    out = RESULTS_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
