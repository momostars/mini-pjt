"""메인 그래프: Router → Planner → Supervisor ↔ 작업 Agent → HITL → Executor → 보고서 (README 3.1절).

그래프 실행에는 config["configurable"]에 thread_id와 toolbox(open_toolbox())가 필요하다.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from src.nodes.context import gather_context
from src.nodes.executor import execute
from src.nodes.extractor import extract
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
    "scheduler": schedule,
    "review": review,
    "executor": execute,
}

RECURSION_LIMIT = 40  # Supervisor 무한 루프 방지 (README 9장)

# 체크포인트에서 복원을 허용할 상태 타입 (등록하지 않은 타입은 향후 차단됨)
STATE_TYPES = [("src.state", name) for name in ("ActionItem", "Decision", "OpenIssue", "EventDraft")]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def _after_router(state: dict) -> str:
    return "planner" if state["intent"] in ("minutes", "schedule", "conflict_check") else "report"


def build_graph(checkpointer=None):
    g = StateGraph(MeetingState)
    g.add_node("router", route)
    g.add_node("planner", plan)
    g.add_node("supervisor", supervise)
    g.add_node("report", report)
    for name, fn in WORKERS.items():
        g.add_node(name, fn)
        g.add_edge(name, "supervisor")

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _after_router, ["planner", "report"])
    g.add_edge("planner", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next_agent"], [*WORKERS, "report"])
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(serde=serializer()))


def run_config(thread_id: str, toolbox, callbacks: list | None = None) -> dict:
    return {
        "configurable": {"thread_id": thread_id, "toolbox": toolbox},
        "callbacks": callbacks or [],
        "recursion_limit": RECURSION_LIMIT,
    }
