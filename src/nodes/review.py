"""HITL 검토 (README 4.4절).

등록 직전에 interrupt()로 멈추고 검토 표를 보여 준다. 응답:
- approve / 승인: 표의 '처리' 열대로 등록 (20건 이상이면 'approve all'로 한 번 더 확인)
- reject / 취소: 아무것도 등록하지 않음
- 그 외 문장: 자연어 수정 요청 → edit 노드가 {id, field, value}로 바꿔 확인받은 뒤 반영
"""

from langgraph.types import interrupt

from src.guardrails import MAX_BULK, scrub
from src.mcp_servers.common import get_employee
from src.state import ActionItem, EventDraft

APPROVE = ("approve", "승인", "y", "yes", "ok", "네", "예")
APPROVE_ALL = ("approve all", "전부 승인", "모두 승인")
REJECT = ("reject", "취소", "거절", "n", "no", "아니오")


def _name(emp_id: str) -> str:
    emp = get_employee(emp_id) or {}
    return f"{emp.get('name', emp_id)} {emp.get('title', '')}({emp.get('team', '')})".strip()


def planned_action(a: ActionItem) -> str:
    if a.excluded:
        return "제외 (사용자 요청)"
    if a.duplicate_of:
        return f"기존 업무 {a.duplicate_of} 기한 변경" if a.due_date else f"기존 업무 {a.duplicate_of} 유지 (등록 안 함)"
    if not a.assignee_id:
        return "등록 안 함 (담당자 미확정)"
    return "신규 등록"


def count_operations(state: dict) -> int:
    n = sum(
        1 for a in state.get("action_items", [])
        if not a.excluded and (a.assignee_id or (a.duplicate_of and a.due_date))
    )
    return n + (1 if state.get("event") else 0)


def _item_row(a: ActionItem) -> dict:
    if a.assignee_id:
        assignee = _name(a.assignee_id)
    elif a.assignee_candidates:
        assignee = "미정 (후보: " + ", ".join(_name(c) for c in a.assignee_candidates) + ")"
    else:
        assignee = "미정"
    due = a.due_date.isoformat() if a.due_date else "없음"
    if a.due_raw:
        due += f" ('{a.due_raw}')"
    return {
        "id": a.id,
        "title": a.title,
        "assignee": assignee,
        "due": due,
        "action": planned_action(a),
        "evidence": a.evidence,
        "warnings": a.warnings,
        "notes": a.notes,
        "needs_confirmation": bool(a.warnings),
    }


def build_review(state: dict) -> dict:
    event: EventDraft | None = state.get("event")
    ops = count_operations(state)
    question = "승인하면 표의 '처리' 열대로 등록합니다. approve / reject / 수정할 내용(예: 'A8 담당자는 박소율로')"
    if ops >= MAX_BULK:
        question = f"등록·변경이 {ops}건으로 많습니다. 정말 등록하려면 'approve all'을 입력하세요. reject / 수정도 가능"
    payload = {
        "type": "review",
        "meeting": state.get("meeting_meta", {}),
        "notices": state.get("guard_notes", []) + ([state["review_notice"]] if state.get("review_notice") else []),
        "items": [_item_row(a) for a in state.get("action_items", [])],
        "event": None if not event else {
            "title": event.title,
            "start": event.start,
            "end": event.end,
            "attendees": [_name(i) for i in event.attendee_ids],
            "warnings": event.warnings,
        },
        "decision_conflicts": [
            {"decision": d.summary, "conflicts_with": d.conflicts_with}
            for d in state.get("decisions", []) if d.conflicts_with
        ],
        "operations": ops,
        "question": question,
    }
    return scrub(payload)  # 출력 가드레일: 검토 표에도 PII가 남지 않게


def review(state: dict) -> dict:
    answer = interrupt(build_review(state))
    text = (answer.get("action") if isinstance(answer, dict) else str(answer or "")).strip()
    low = text.lower()
    update = {"done": ["review"], "review_notice": None}

    if low in APPROVE_ALL:
        return {**update, "review_status": "approved", "bulk_confirmed": True}
    if low in APPROVE:
        return {**update, "review_status": "approved"}
    if not text:  # 빈 응답은 거절로 보지 않고 다시 묻는다
        required = {**(state.get("required") or {}), "review": state.get("done", []).count("review") + 2}
        return {**update, "review_status": "pending", "required": required,
                "review_notice": "응답이 없어 다시 확인합니다."}
    if low in REJECT:
        return {**update, "review_status": "rejected"}
    return {**update, "review_status": "edited", "edit_request": text, "pending_edits": None}
