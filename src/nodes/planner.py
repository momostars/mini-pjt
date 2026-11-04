"""Planner: 의도에 맞춰 처리 단계를 정한다 (README 3.2절).

지금은 규칙 기반. 회의록 유형에 따라 단계를 바꿀 필요가 생기면 LLM 계획으로 확장한다.
"""

from src.nodes.common import detect_meeting_date

PLANS = {
    "minutes": ["extractor", "context", "resolver", "review", "executor", "report"],
    "conflict_check": ["extractor", "context", "report"],
    "schedule": ["scheduler", "review", "executor", "report"],
}


def plan(state: dict) -> dict:
    meta = dict(state.get("meeting_meta") or {})
    meta["date"] = detect_meeting_date(state.get("masked_minutes", ""))
    return {"plan": PLANS[state["intent"]], "meeting_meta": meta, "done": []}
