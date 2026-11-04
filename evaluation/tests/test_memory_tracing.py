"""M6 테스트 (LLM 없음): trace.jsonl 기록 규칙과 집계, 장기 기억 저장."""

import asyncio
import json
from uuid import uuid4

from langgraph.store.memory import InMemoryStore

from src.memory import get_alias, get_glossary, get_prefs, learn_alias
from src.tracing import JsonlTraceHandler, load_events, summarize, tools_called


class GraphInterrupt(Exception):  # LangGraph 인터럽트와 같은 이름 (판정은 이름으로 함)
    pass


def _handler(tmp_path):
    return JsonlTraceHandler(tmp_path / "trace.jsonl", trace_id="t1")


def test_node_events_only_for_graph_nodes(tmp_path):
    h = _handler(tmp_path)
    node, inner = uuid4(), uuid4()
    h.on_chain_start({}, {}, run_id=node, metadata={"langgraph_node": "resolver"}, name="resolver")
    h.on_chain_start({}, {}, run_id=inner, parent_run_id=node, metadata={"langgraph_node": "resolver"},
                     name="RunnableSequence")  # 노드 안 내부 체인
    h.on_chain_end({}, run_id=inner)
    h.on_chain_end({}, run_id=node)
    events = load_events(h.path)
    assert [e["event"] for e in events] == ["node_start", "node_end"]
    assert events[1]["node"] == "resolver" and events[1]["latency_ms"] >= 0
    assert all(e["trace_id"] == "t1" and e["ts"].endswith("+09:00") for e in events)


def test_interrupt_is_not_error(tmp_path):
    h = _handler(tmp_path)
    run = uuid4()
    h.on_chain_start({}, {}, run_id=run, metadata={"langgraph_node": "review"}, name="review")
    h.on_chain_error(GraphInterrupt(), run_id=run)
    assert load_events(h.path)[-1]["event"] == "interrupt"


def test_tool_args_masked_and_clipped(tmp_path):
    h = _handler(tmp_path)
    run = uuid4()
    args = {"title": "연락 010-2345-6789, 850315-1234567 " + "가" * 600}
    h.on_tool_start({"name": "create_task"}, str(args), run_id=run, inputs=args, name="create_task")
    h.on_tool_end([{"type": "text", "text": '{"id": "T-035"}'}], run_id=run)
    text = h.path.read_text(encoding="utf-8")
    assert "010-2345-6789" not in text and "850315-1234567" not in text
    end = load_events(h.path)[-1]
    assert end["event"] == "tool_end" and len(end["args"]) <= 501
    assert end["output"] == '{"id": "T-035"}'


def test_llm_tokens_without_prompt(tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    h = _handler(tmp_path)
    run = uuid4()
    h.on_chat_model_start({}, [[]], run_id=run, metadata={"ls_model_name": "m", "langgraph_node": "router"})
    msg = AIMessage(content="비밀 응답 본문", usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13})
    h.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=run)
    text = h.path.read_text(encoding="utf-8")
    assert "비밀 응답 본문" not in text
    end = load_events(h.path)[-1]
    assert (end["input_tokens"], end["output_tokens"], end["model"]) == (10, 3, "m")


def test_tool_error_and_summary(tmp_path):
    h = _handler(tmp_path)
    ok, bad = uuid4(), uuid4()
    h.on_tool_start({}, "", run_id=ok, inputs={}, name="find_employee")
    h.on_tool_end("{}", run_id=ok)
    h.on_tool_start({}, "", run_id=bad, inputs={}, name="create_task")
    h.on_tool_error(ValueError("휴직 중"), run_id=bad)
    events = load_events(h.path)
    assert tools_called(events) == {"find_employee"}  # 실패한 호출은 '호출됨'으로 치지 않는다
    s = summarize(events)["t1"]
    assert s["tool_ok"]["find_employee"] == 1 and s["tool_err"]["create_task"] == 1


def test_handler_never_raises(tmp_path):
    h = _handler(tmp_path)
    h.on_tool_end("x", run_id=uuid4())  # 시작 기록 없는 종료
    h.on_llm_end(None, run_id=uuid4())  # 잘못된 응답
    h.on_chain_error(RuntimeError(), run_id=uuid4())


# ── 장기 기억 ──────────────────────────────────────────


def test_alias_learning_counts_and_user_separation():
    async def run():
        store = InMemoryStore()
        a = {"configurable": {"user_id": "alice"}}
        b = {"configurable": {"user_id": "bob"}}
        await learn_alias(store, a, "QA 쪽", "E019", "박소율 선임(QA팀)")
        await learn_alias(store, a, "QA쪽", "E019", "박소율 선임(QA팀)")  # 공백 차이는 같은 호칭
        got = await get_alias(store, a, "QA 쪽")
        assert got["employee_id"] == "E019" and got["count"] == 2
        assert await get_alias(store, b, "QA 쪽") is None  # 사용자별로 분리
        await learn_alias(store, a, "QA 쪽", "E018", "김민준 선임(QA팀)")  # 다른 사람으로 고치면 횟수 초기화
        assert (await get_alias(store, a, "QA 쪽"))["count"] == 1

    asyncio.run(run())


def test_no_store_means_no_memory():
    async def run():
        assert await get_alias(None, {}, "QA 쪽") is None
        assert await get_prefs(None, {}) == {} and await get_glossary(None, {}) == {}
        await learn_alias(None, {}, "QA 쪽", "E019", "x")  # 오류 없이 무시

    asyncio.run(run())
