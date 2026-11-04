"""요약 보고서 (README 3.1절 마지막 단계). 코드로 만든다(LLM 없음)."""

from langchain_core.messages import AIMessage

from src.nodes.review import _item_row


def _minutes_report(state: dict) -> list[str]:
    lines = []
    meta = state.get("meeting_meta") or {}
    lines.append(f"# 회의 정리: {meta.get('title') or '회의'} ({meta.get('date') or '날짜 미상'})")

    decisions = state.get("decisions", [])
    lines.append(f"\n## 결정 사항 {len(decisions)}건")
    for d in decisions:
        lines.append(f"- {d.summary}")
        for c in d.conflicts_with:
            lines.append(f"  - ⚠ 이전 결정과 충돌: {c}")

    items = state.get("action_items", [])
    lines.append(f"\n## 할 일 {len(items)}건")
    if not items:
        lines.append("- 회의록에서 할 일을 찾지 못했습니다.")
    for a in items:
        row = _item_row(a)
        lines.append(f"- [{a.id}] {a.title} | 담당 {row['assignee']} | 기한 {row['due']} | {row['action']}")
        for w in a.warnings:
            lines.append(f"  - ⚠ {w}")
        for n in a.notes:
            lines.append(f"  - {n}")

    unresolved = [a for a in items if not a.assignee_id and not a.duplicate_of]
    if unresolved:
        lines.append("\n## 미확정 항목 (등록하지 않음)")
        lines += [f"- [{a.id}] {a.title}: 담당자 지정 필요" for a in unresolved]

    issues = state.get("open_issues", [])
    if issues:
        lines.append("\n## 미결 이슈")
        lines += [f"- {i.summary}" + (f" → {i.follow_up}" if i.follow_up else "") for i in issues]
    return lines


def _conflict_report(state: dict) -> list[str]:
    lines = ["# 이전 결정과의 충돌 확인"]
    decisions = state.get("decisions", [])
    if not decisions:
        return lines + ["- 확인할 결정 사항을 찾지 못했습니다."]
    for d in decisions:
        lines.append(f"\n## {d.summary}")
        if d.conflicts_with:
            lines += [f"- ⚠ 충돌: {c}" for c in d.conflicts_with]
        else:
            lines.append("- 충돌하는 이전 결정을 찾지 못했습니다.")
    lines.append("\n(등록은 하지 않았습니다.)")
    return lines


def _event_report(state: dict) -> list[str]:
    e = state.get("event")
    if not e:
        return ["# 일정 등록", "- 일정 정보를 찾지 못했습니다."]
    lines = ["# 일정 등록", f"- {e.title} | {e.start or e.date_raw} ~ {e.end or ''} | 참석 {len(e.attendee_ids)}명"]
    lines += [f"  - ⚠ {w}" for w in e.warnings]
    return lines


def report(state: dict) -> dict:
    intent = state.get("intent")
    if state.get("report") and intent in ("out_of_scope", "empty"):
        text = state["report"]
    else:
        if intent == "conflict_check":
            lines = _conflict_report(state)
        elif intent == "schedule":
            lines = _event_report(state)
        else:
            lines = _minutes_report(state)

        status = state.get("review_status")
        if status == "rejected":
            lines.append("\n## 결과\n- 사용자가 거절해 아무것도 등록하지 않았습니다.")
        elif status == "approved":
            lines.append("\n## 결과")
            lines += [f"- ✅ {r}" for r in state.get("registered", [])] or ["- 등록·변경된 항목이 없습니다."]
            lines += [f"- ❌ {f}" for f in state.get("failed", [])]
        elif intent in ("minutes", "schedule") and (state.get("action_items") or state.get("event")):
            lines.append("\n## 결과\n- 승인 전이라 등록하지 않았습니다.")
        text = "\n".join(lines)

    return {"report": text, "messages": [AIMessage(content=text)], "done": ["report"]}
