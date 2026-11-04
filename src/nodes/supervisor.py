"""Supervisor: 다음 작업자를 고른다 (README 3.2절).

- 기본: 계획(plan) 순서대로. 단계마다 필요한 실행 횟수(required)보다 덜 실행됐으면 그 단계를 고른다
- 등록할 것이 없으면 검토·등록을 건너뛰고, 승인되지 않았으면 Executor로 보내지 않는다
- HITL 수정 요청이면 edit → confirm_edit → (검토 다시)
- 출력 검증(validate)에 실패하면 LLM이 다시 실행할 노드를 고른다 (최대 MAX_RETRIES회)
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from src.llm import get_llm

MAX_RETRIES = 1


class RerunChoice(BaseModel):
    node: Literal["extractor", "resolver", "none"]
    reason: str


RERUN_SYSTEM = """할 일 출력 검증에서 오류가 났습니다. 어느 단계를 다시 실행할지 고릅니다.
- extractor: 근거 문장이 회의록에 없는 등 추출 자체가 잘못된 경우
- resolver: 담당자 ID나 기한 날짜가 잘못된 경우
- none: 다시 실행해도 나아지지 않을 오류(이미 항목이 제외됨 등). 사용자 검토로 넘깁니다."""


def needs(state: dict, step: str) -> bool:
    return state.get("done", []).count(step) < (state.get("required") or {}).get(step, 1)


def _nothing_to_register(state: dict) -> bool:
    return not state.get("action_items") and not state.get("event")


async def _choose_rerun(errors: list[str], config: RunnableConfig) -> RerunChoice:
    prompt = ChatPromptTemplate.from_messages([("system", RERUN_SYSTEM), ("human", "{errors}")])
    return await (prompt | get_llm().with_structured_output(RerunChoice)).ainvoke(
        {"errors": "\n".join(errors)}, config=config
    )


async def supervise(state: dict, config: RunnableConfig) -> dict:
    plan = state.get("plan", [])

    # HITL 수정 흐름
    if state.get("review_status") == "edited":
        return {"next_agent": "confirm_edit" if state.get("pending_edits") else "edit"}

    # 출력 검증 실패 → 재실행 판단
    errors = state.get("validation_errors") or []
    retries = state.get("retries", 0)
    if errors and retries < MAX_RETRIES and "validate" in plan and not needs(state, "validate"):
        choice = await _choose_rerun(errors, config)
        if choice.node != "none":
            done, required = state.get("done", []), dict(state.get("required") or {})
            for step in plan[plan.index(choice.node): plan.index("validate") + 1]:
                required[step] = done.count(step) + 1
            return {"next_agent": choice.node, "required": required, "retries": retries + 1, "validation_errors": []}
        return {"retries": retries + 1, "validation_errors": [], "next_agent": _next_step(state)}

    return {"next_agent": _next_step(state)}


def _next_step(state: dict) -> str:
    for step in state.get("plan", []):
        if not needs(state, step):
            continue
        if step in ("review", "executor") and _nothing_to_register(state):
            continue
        if step == "executor" and state.get("review_status") != "approved":
            continue
        return step
    return "report"
