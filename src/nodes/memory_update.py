"""장기 기억 갱신 (README 3.1절 '장기 기억 갱신', 4.6절).

HITL에서 사용자가 담당자를 고쳤으면 "원문 호칭 → 직원"을 저장한다.
승인·거절과 관계없이 저장한다. 사용자가 직접 확인한 수정이기 때문이다.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from src.mcp_servers.common import get_employee
from src.memory import learn_alias
from src.nodes.common import employee_label


async def update_memory(state: dict, config: RunnableConfig, *, store: BaseStore | None = None) -> dict:
    items = {a.id: a for a in state.get("action_items", [])}
    learned = []
    for edit in state.get("edits_applied") or []:
        item = items.get(edit["item_id"])
        if edit["field"] != "assignee" or not item or not item.assignee_raw.strip():
            continue
        emp = get_employee(edit["assignee_id"])
        if not emp:
            continue
        await learn_alias(store, config, item.assignee_raw, emp["id"], employee_label(emp))
        learned.append(f"호칭 '{item.assignee_raw}' → {employee_label(emp)}")
    return {"learned": learned, "done": ["memory"]}
