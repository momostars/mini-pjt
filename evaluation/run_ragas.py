"""RAG 평가 (README 5.3절).

지금은 LLM 없이 측정하는 Hit@k와 MRR만 있다. RAGAS(Context Precision/Recall, Faithfulness)는
Context Agent의 경고 문구가 필요해 M7에서 추가한다.

    python -m evaluation.run_ragas               # 유사도 검색
    python -m evaluation.run_ragas --mmr         # MMR 비교
    python -m evaluation.run_ragas -k 4 --name chunk_section

먼저 python -m src.retriever --ingest 로 인덱스를 만들어야 한다.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.retriever import DEFAULT_K, search_past_meetings

DATASET = Path("evaluation/dataset/retrieval.jsonl")
RESULTS_DIR = Path("evaluation/results")

TARGET_HIT = 0.90
TARGET_MRR = 0.75


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


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 평가 (Hit@k, MRR)")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--mmr", action="store_true")
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or ("mmr" if args.mmr else "similarity")
    out = RESULTS_DIR / f"hitk_{datetime.now():%Y%m%d_%H%M%S}_{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
