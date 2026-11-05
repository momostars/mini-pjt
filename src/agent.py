"""메인 그래프 (README 3.1절).

입력 가드레일 → Router → Planner → Supervisor ↔ 작업 Agent → 출력 검증 → HITL 검토(수정) → Executor → 보고서

그래프 실행에는 config["configurable"]에 thread_id와 toolbox(open_toolbox())가 필요하다.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from src.nodes.context import gather_context
from src.nodes.edit import confirm_edit, edit
from src.nodes.executor import execute
from src.nodes.extractor import extract
from src.nodes.guard import guard_input, validate
from src.nodes.memory_update import update_memory
from src.nodes.planner import plan
from src.nodes.report import report
from src.nodes.resolver import resolve
from src.nodes.review import review
from src.nodes.router import route
from src.nodes.scheduler import schedule
from src.nodes.supervisor import supervise
from src.state import MeetingState

WORKERS = {
    "extractor": extract,
    "context": gather_context,
    "resolver": resolve,
    "validate": validate,
    "scheduler": schedule,
    "review": review,
    "edit": edit,
    "confirm_edit": confirm_edit,
    "executor": execute,
    "memory": update_memory,
}

RECURSION_LIMIT = 60  # Supervisor 무한 루프 방지 (README 9장). 수정 반복을 고려해 여유를 둠

# 체크포인트에서 복원을 허용할 상태 타입 (등록하지 않은 타입은 향후 차단됨)
STATE_TYPES = [("src.state", name) for name in ("ActionItem", "Decision", "OpenIssue", "EventDraft")]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def _after_guard(state: dict) -> str:
    return "report" if state.get("blocked") else "router"


def _after_router(state: dict) -> str:
    return "planner" if state["intent"] in ("minutes", "schedule", "conflict_check") else "report"


def build_graph(checkpointer=None, store=None):
    g = StateGraph(MeetingState)
    g.add_node("guard_input", guard_input)
    g.add_node("router", route)
    g.add_node("planner", plan)
    g.add_node("supervisor", supervise)
    g.add_node("report", report)
    for name, fn in WORKERS.items():
        g.add_node(name, fn)
        g.add_edge(name, "supervisor")

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", _after_guard, ["router", "report"])
    g.add_conditional_edges("router", _after_router, ["planner", "report"])
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next_agent"], [*WORKERS, "report"])
    g.add_edge("report", END)
    # store가 없으면 장기 기억 없이 동작한다
    # (평가 러너는 실행마다 새 SQLite Store를 만들고 케이스마다 user_id를 다르게 줘서 기억을 분리한다)
    return g.compile(checkpointer=checkpointer or InMemorySaver(serde=serializer()), store=store)


@asynccontextmanager
async def sqlite_checkpointer(path: str | None = None):
    """중단된 실행(검토 대기)을 프로그램을 다시 켜도 이어갈 수 있게 SQLite에 저장한다."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = path or os.getenv("CHECKPOINT_DB", "./.state/checkpoints.sqlite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        saver.serde = serializer()
        yield saver


def run_config(thread_id: str, toolbox, callbacks: list | None = None, user_id: str | None = None,
               trace_path: str | None = None, trace: bool = True) -> dict:
    """그래프 실행 config. trace=True면 trace.jsonl 기록 핸들러(trace_id = thread_id)를 붙인다."""
    from src.tracing import JsonlTraceHandler

    handlers = list(callbacks or [])
    if trace:
        handlers.append(JsonlTraceHandler(trace_path, trace_id=thread_id))
    configurable = {"thread_id": thread_id, "toolbox": toolbox}
    if user_id:
        configurable["user_id"] = user_id
    return {"configurable": configurable, "callbacks": handlers, "recursion_limit": RECURSION_LIMIT}
