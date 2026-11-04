"""Supervisor: 계획 순서대로 다음 작업자를 고른다 (README 3.2절).

등록할 것이 없으면 검토·등록을 건너뛰고, 승인되지 않았으면 등록하지 않는다.
출력 검증 실패 시 LLM으로 재실행 노드를 고르는 부분은 M5(출력 가드레일)에서 추가한다.
"""


def _nothing_to_register(state: dict) -> bool:
    return not state.get("action_items") and not state.get("event")


def supervise(state: dict) -> dict:
    done = set(state.get("done", []))
    for step in state.get("plan", []):
        if step in done:
            continue
        if step in ("review", "executor") and _nothing_to_register(state):
            continue
        if step == "executor" and state.get("review_status") != "approved":
            continue
        return {"next_agent": step}
    return {"next_agent": "report"}
