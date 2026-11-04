"""Executor Agent: 승인된 항목만 등록·변경한다 (README 4.4절 승인 시 처리 규칙).

- 담당자가 확정되지 않은 항목은 등록하지 않는다(보고서의 미확정 항목)
- 기한이 없으면 비워 두고 등록한다
- 중복 의심 항목은 신규 등록하지 않고, 기한이 있으면 update_task로 기존 업무 기한을 바꾼다
"""

from langchain_core.runnables import RunnableConfig

from src.guardrails import MAX_BULK
from src.nodes.common import toolbox
from src.nodes.review import count_operations
from src.state import ActionItem, EventDraft
from src.tools import ToolCallError


async def execute(state: dict, config: RunnableConfig) -> dict:
    if state.get("review_status") != "approved":  # 승인 없이 등록하지 않는다 (이중 확인)
        return {"done": ["executor"]}

    ops = count_operations(state)
    if ops >= MAX_BULK and not state.get("bulk_confirmed"):  # 실행 가드레일: 대량 등록은 따로 확인
        return {"failed": [f"등록·변경이 {ops}건이라 멈췄습니다. 'approve all'로 다시 승인해 주세요."],
                "done": ["executor"]}

    tb = toolbox(config)
    source = (state.get("meeting_meta") or {}).get("source")
    registered, failed = [], []

    event: EventDraft | None = state.get("event")
    if event:
        if event.start and event.end and event.attendee_ids:
            try:
                created = await tb.call(
                    "create_event",
                    {"title": event.title, "start": event.start, "end": event.end, "attendees": event.attendee_ids},
                    config,
                )
                registered.append(f"{created['id']} 일정 '{event.title}' 등록")
            except ToolCallError as e:
                failed.append(f"일정 '{event.title}': {e}")
        else:
            failed.append(f"일정 '{event.title}': 날짜·시각·참석자가 확정되지 않아 등록하지 않음")

    item: ActionItem
    for item in state.get("action_items", []):
        if item.excluded:
            continue
        try:
            if item.duplicate_of:
                if item.due_date:
                    r = await tb.call(
                        "update_task", {"task_id": item.duplicate_of, "due_date": item.due_date.isoformat()}, config
                    )
                    if r["before"] != r["after"]:
                        registered.append(f"{item.duplicate_of} 기한 변경 {r['before']} → {r['after']} ({item.title})")
                    else:
                        registered.append(f"{item.duplicate_of} 기존 업무와 기한 같음, 변경 없음 ({item.title})")
                continue
            if not item.assignee_id:
                continue
            created = await tb.call(
                "create_task",
                {
                    "title": item.title,
                    "assignee_id": item.assignee_id,
                    "due_date": item.due_date.isoformat() if item.due_date else None,
                    "source": source,
                },
                config,
            )
            registered.append(f"{created['id']} '{item.title}' 등록")
        except ToolCallError as e:
            failed.append(f"[{item.id}] {item.title}: {e}")

    return {"registered": registered, "failed": failed, "done": ["executor"]}
