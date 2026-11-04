"""Planner: 의도에 맞춰 처리 단계를 정한다 (README 3.2절).

지금은 규칙 기반. 회의록 유형에 따라 단계를 바꿀 필요가 생기면 LLM 계획으로 확장한다.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from src.memory import get_glossary
from src.nodes.common import detect_meeting_date

PLANS = {
    "minutes": ["extractor", "context", "resolver", "validate", "review", "executor", "memory", "report"],
    "conflict_check": ["extractor", "context", "report"],
    "schedule": ["scheduler", "review", "executor", "report"],
}


async def plan(state: dict, config: RunnableConfig, *, store: BaseStore | None = None) -> dict:
    meta = dict(state.get("meeting_meta") or {})
    meta["date"] = detect_meeting_date(state.get("masked_minutes", ""))
    return {"plan": PLANS[state["intent"]], "meeting_meta": meta, "glossary": await get_glossary(store, config)}
