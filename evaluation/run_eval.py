"""기능 지표 평가 (README 5.2절, SERVICE.md 5절).

    python -m evaluation.run_eval                 # 추출 + 라우팅
    python -m evaluation.run_eval --only extract  # 추출 지표만
    python -m evaluation.run_eval --only routing  # Router 의도 분류만

1. 추출 지표: evaluation/dataset/meetings/gold.json의 회의록마다 그래프를 검토(HITL) 직전까지 실행하고
   검토 표에 올라온 할 일을 정답과 비교한다. 승인하지 않으므로 아무것도 등록하지 않는다.
   - 할 일 짝짓기: LLM이 제목·근거 문장의 의미로 정답 할 일과 1:1로 짝짓는다 (README 5.2절)
   - Recall = 짝지은 수 / 정답 수, Precision = 짝지은 수 / 추출 수
   - 담당자·기한 정확도: 짝지은 할 일에서 값이 정확히 같은지 코드로 비교
     (정답이 팀 단위 후보면 담당자를 확정하지 않고 후보에 정답 인원이 모두 들어 있어야 정답)
   - HITL 발생률: 정답에 needs_hitl로 표시한 할 일에 경고가 붙어 '확인 필요'로 올라왔는지
   - 근거 없는 할 일: 근거 문장이 (마스킹한) 회의록 원문에 없는 할 일 수
2. Router 의도 분류 정확도: test_queries.csv 20건의 입력을 입력 가드레일 → Router만 통과시켜
   evaluation/dataset/routing.jsonl의 정답 의도와 비교한다.

결과: evaluation/results/eval_<실행시각>/ (summary.json, items.csv, routing.csv, trace.jsonl)
"""

import argparse
import asyncio
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agent import build_graph, run_config, sqlite_checkpointer
from src.llm import MODEL, get_llm
from src.mcp_servers.common import reset_calendar, reset_tasks
from src.memory import sqlite_store
from src.nodes.guard import guard_input
from src.nodes.router import REFUSALS, route
from src.tools import open_toolbox

GOLD = Path("evaluation/dataset/meetings/gold.json")
ROUTING = Path("evaluation/dataset/routing.jsonl")
CASES_CSV = Path("evaluation/test_queries.csv")
RESULTS_DIR = Path("evaluation/results")

TARGETS = {
    "recall": 0.85, "precision": 0.85, "assignee_accuracy": 0.90, "due_accuracy": 0.90,
    "hitl_rate": 1.0, "router_accuracy": 0.90, "ungrounded": 0,
}


# ── 할 일 짝짓기 (LLM-as-judge) ───────────────────────────


class Pair(BaseModel):
    gold_no: int = Field(description="정답 할 일 번호")
    pred_id: str | None = Field(description="같은 할 일로 판단한 추출 항목 id. 없으면 null")
    reason: str = Field(description="짝지은 이유 한 문장")


class Matching(BaseModel):
    pairs: list[Pair]


MATCH_SYSTEM = """회의록에서 추출한 할 일 목록을 정답 목록과 비교합니다.
정답 할 일마다 같은 일을 가리키는 추출 항목을 하나 고르세요.

- 제목 문구가 달라도 같은 업무(같은 대상에 같은 행동)면 같은 할 일입니다. 근거 문장도 참고하세요.
- 담당자나 기한이 달라도 업무가 같으면 짝지어도 됩니다 (담당자·기한은 따로 채점합니다).
- 1:1로만 짝짓습니다. 추출 항목 하나를 정답 두 개에 쓰지 않습니다.
- 맞는 항목이 없으면 pred_id를 null로 둡니다.
- 정답 할 일은 빠짐없이 모두 한 번씩 답합니다."""

MATCH_HUMAN = """## 정답 할 일
{gold}

## 추출한 할 일
{pred}"""


async def match_items(gold: list[dict], pred: list[dict]) -> dict[int, str | None]:
    if not gold:
        return {}
    if not pred:
        return {g["no"]: None for g in gold}
    prompt = ChatPromptTemplate.from_messages([("system", MATCH_SYSTEM), ("human", MATCH_HUMAN)])
    chain = prompt | get_llm().with_structured_output(Matching)
    result: Matching = await chain.ainvoke({
        "gold": "\n".join(f"{g['no']}. {g['title']}" for g in gold),
        "pred": "\n".join(f"- id={p['id']} | {p['title']} | 근거: {p['evidence']}" for p in pred),
    })
    pred_ids = {p["id"] for p in pred}
    matched, used = {}, set()
    for pair in result.pairs:  # 없는 id, 중복 사용은 짝짓지 않은 것으로 본다
        pid = pair.pred_id if pair.pred_id in pred_ids and pair.pred_id not in used else None
        if pid:
            used.add(pid)
        matched[pair.gold_no] = pid
    return {g["no"]: matched.get(g["no"]) for g in gold}


# ── 추출 지표 ─────────────────────────────────────────────


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def dump(obj):
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def assignee_ok(gold: dict, pred: dict) -> bool:
    if gold.get("candidates"):  # 팀 단위: 확정하지 않고 후보에 정답 인원이 모두 있어야 함
        return pred.get("assignee_id") is None and set(gold["candidates"]) <= set(pred.get("assignee_candidates") or [])
    return pred.get("assignee_id") == gold["assignee"]


async def run_meeting(meeting: dict, graph, toolbox, trace_path: Path, stamp: str, n: int) -> dict:
    """검토 표가 나올 때까지(또는 끝날 때까지) 실행하고 그 시점의 할 일을 돌려준다."""
    thread_id = f"ev{n:02d}-{stamp}"
    reset_tasks()
    reset_calendar()
    config = run_config(thread_id, toolbox, user_id=f"eval-{thread_id}", trace_path=str(trace_path))
    t0 = time.perf_counter()
    await graph.ainvoke({"request": f"{meeting['file']} 회의록 정리해줘"}, config)
    elapsed = round(time.perf_counter() - t0, 1)
    state = (await graph.aget_state(config)).values
    items = [d for d in map(dump, state.get("action_items") or []) if not d.get("excluded")]
    return {"elapsed_s": elapsed, "items": items, "minutes": state.get("masked_minutes") or ""}


async def eval_extraction(out_dir: Path, stamp: str) -> tuple[dict, list[dict]]:
    meetings = json.loads(GOLD.read_text(encoding="utf-8"))["meetings"]
    rows, per_meeting = [], []
    totals = {"gold": 0, "pred": 0, "matched": 0, "assignee_ok": 0, "due_ok": 0,
              "hitl_need": 0, "hitl_hit": 0, "ungrounded": 0}

    async with sqlite_checkpointer(str(out_dir / "checkpoints.sqlite")) as saver, \
            sqlite_store(str(out_dir / "memory.sqlite")) as store, open_toolbox() as tb:
        graph = build_graph(saver, store)
        for n, meeting in enumerate(meetings, 1):
            run = await run_meeting(meeting, graph, tb, out_dir / "trace.jsonl", stamp, n)
            gold, pred = meeting["action_items"], run["items"]
            pairs = await match_items(gold, pred)
            by_id = {p["id"]: p for p in pred}
            minutes = normalize(run["minutes"])
            ungrounded = [p["id"] for p in pred if normalize(p["evidence"]) not in minutes]

            m = {"matched": 0, "assignee_ok": 0, "due_ok": 0}
            for g in gold:
                p = by_id.get(pairs.get(g["no"]))
                a_ok = d_ok = hitl = None
                if p:
                    m["matched"] += 1
                    a_ok = assignee_ok(g, p)
                    d_ok = p.get("due_date") == g["due"]
                    m["assignee_ok"] += a_ok
                    m["due_ok"] += d_ok
                if g["needs_hitl"]:
                    totals["hitl_need"] += 1
                    hitl = bool(p and p.get("warnings"))
                    totals["hitl_hit"] += hitl
                rows.append({
                    "meeting": Path(meeting["file"]).name, "gold_no": g["no"], "gold_title": g["title"],
                    "pred_id": p["id"] if p else "", "pred_title": p["title"] if p else "",
                    "gold_assignee": g["assignee"] or "/".join(g.get("candidates", [])),
                    "pred_assignee": (p.get("assignee_id") or "후보:" + "/".join(p.get("assignee_candidates") or []))
                    if p else "",
                    "assignee_ok": a_ok, "gold_due": g["due"], "pred_due": p.get("due_date") if p else "",
                    "due_ok": d_ok, "needs_hitl": g["needs_hitl"], "hitl_ok": hitl,
                })
            matched_ids = {pid for pid in pairs.values() if pid}
            for p in pred:  # 정답에 없는 추출 (Precision 감점)
                if p["id"] not in matched_ids:
                    rows.append({"meeting": Path(meeting["file"]).name, "gold_no": "", "gold_title": "(정답 없음)",
                                 "pred_id": p["id"], "pred_title": p["title"],
                                 "pred_assignee": p.get("assignee_id") or "", "pred_due": p.get("due_date") or ""})

            totals["gold"] += len(gold)
            totals["pred"] += len(pred)
            totals["ungrounded"] += len(ungrounded)
            for k in m:
                totals[k] += m[k]
            per_meeting.append({"meeting": Path(meeting["file"]).name, "gold": len(gold), "pred": len(pred),
                                **m, "ungrounded": ungrounded, "elapsed_s": run["elapsed_s"]})
            print(f"  {Path(meeting['file']).name:<30} 정답 {len(gold)} / 추출 {len(pred)} / 짝 {m['matched']}"
                  f" / 담당 {m['assignee_ok']} / 기한 {m['due_ok']} / 근거 없음 {len(ungrounded)}"
                  f"  ({run['elapsed_s']}s)")

    t = totals
    metrics = {
        "recall": round(t["matched"] / t["gold"], 3) if t["gold"] else None,
        "precision": round(t["matched"] / t["pred"], 3) if t["pred"] else None,
        "assignee_accuracy": round(t["assignee_ok"] / t["matched"], 3) if t["matched"] else None,
        "due_accuracy": round(t["due_ok"] / t["matched"], 3) if t["matched"] else None,
        "hitl_rate": round(t["hitl_hit"] / t["hitl_need"], 3) if t["hitl_need"] else None,
        "ungrounded": t["ungrounded"],
        "avg_elapsed_to_review_s": round(sum(m["elapsed_s"] for m in per_meeting) / len(per_meeting), 1),
        "counts": t,
        "per_meeting": per_meeting,
    }
    return metrics, rows


# ── Router 의도 분류 ──────────────────────────────────────


def refusal_of(report: str) -> str | None:
    return next((k for k, msg in REFUSALS.items() if (report or "").startswith(msg)), None)


async def eval_routing() -> tuple[dict, list[dict]]:
    with CASES_CSV.open(encoding="utf-8-sig") as f:
        inputs = {int(r["id"]): r["input"] for r in csv.DictReader(f)}
    gold = [json.loads(line) for line in ROUTING.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for g in gold:
        state = {"request": inputs[g["id"]]}
        state |= await guard_input(state, {})
        if not state.get("blocked"):
            state |= await route(state, {})
        intent = state.get("intent")
        refusal = refusal_of(state.get("report")) if intent == "out_of_scope" else None
        ok = intent == g["intent"] and (g["intent"] != "out_of_scope" or refusal == g.get("refusal"))
        rows.append({"id": g["id"], "gold_intent": g["intent"], "pred_intent": intent,
                     "gold_refusal": g.get("refusal", ""), "pred_refusal": refusal or "",
                     "intent_ok": intent == g["intent"], "ok": ok})
        if not ok:
            print(f"  #{g['id']} 정답 {g['intent']}/{g.get('refusal', '-')} → 예측 {intent}/{refusal or '-'}")
    n = len(rows)
    metrics = {
        "router_accuracy": round(sum(r["intent_ok"] for r in rows) / n, 3),
        "router_accuracy_with_refusal": round(sum(r["ok"] for r in rows) / n, 3),
        "cases": n,
    }
    return metrics, rows


# ── 실행 ──────────────────────────────────────────────────


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verdict(name: str, value) -> str:
    target = TARGETS[name]
    if value is None:
        return "-"
    ok = value <= target if name == "ungrounded" else value >= target
    return "충족" if ok else "미달"


async def main_async(only: str | None) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"run_at": stamp, "model": MODEL}

    if only in (None, "extract"):
        print("추출 지표 (회의록 → 검토 표)")
        metrics, rows = await eval_extraction(out_dir, stamp)
        summary["extraction"] = metrics
        write_csv(out_dir / "items.csv", rows)
    if only in (None, "routing"):
        print("\nRouter 의도 분류")
        metrics, rows = await eval_routing()
        summary["routing"] = metrics
        write_csv(out_dir / "routing.csv", rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| 지표 | 목표 | 결과 | 판정 |")
    print("|---|---|---|---|")
    flat = {**summary.get("extraction", {}), **summary.get("routing", {})}
    for name, target in TARGETS.items():
        if name in flat:
            print(f"| {name} | {target} | {flat[name]} | {verdict(name, flat[name])} |")
    print(f"\n결과: {out_dir}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="기능 지표 평가")
    parser.add_argument("--only", choices=["extract", "routing"], help="한쪽만 실행")
    args = parser.parse_args()
    asyncio.run(main_async(args.only))


if __name__ == "__main__":
    main()
