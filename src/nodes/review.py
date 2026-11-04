"""HITL 검토 (README 4.4절). M4는 approve/reject만 받는 최소 버전.

등록 직전에 interrupt()로 멈추고 검토 표를 보여 준다. 자연어 수정과 SqliteSaver 재개는 M5에서 추가한다.
"""

from langgraph.types import interrupt

from src.mcp_servers.common import get_employee
from src.state import ActionItem, EventDraft


def _name(emp_id: str) -> str:
    emp = get_employee(emp_id) or {}
    return f"{emp.get('name', emp_id)} {emp.get('title', '')}({emp.get('team', '')})".strip()


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
    if a.duplicate_of:
        action = f"기존 업무 {a.duplicate_of} 기한 변경" if a.due_date else f"기존 업무 {a.duplicate_of} 유지 (등록 안 함)"
    elif not a.assignee_id:
        action = "등록 안 함 (담당자 미확정)"
    else:
        action = "신규 등록"
    return {
        "id": a.id,
        "title": a.title,
        "assignee": assignee,
        "due": due,
        "action": action,
        "evidence": a.evidence,
        "warnings": a.warnings,
        "notes": a.notes,
        "needs_confirmation": bool(a.warnings),
    }


def build_review(state: dict) -> dict:
    event: EventDraft | None = state.get("event")
    return {
        "meeting": state.get("meeting_meta", {}),
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
        "question": "승인하면 표의 '처리' 열대로 등록합니다. approve / reject",
    }


def review(state: dict) -> dict:
    answer = interrupt(build_review(state))
    action = answer.get("action") if isinstance(answer, dict) else str(answer)
    approved = (action or "").strip().lower() in ("approve", "승인", "y", "yes")
    return {"review_status": "approved" if approved else "rejected", "done": ["review"]}
