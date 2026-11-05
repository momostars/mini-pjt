"""장애 주입 테스트 (README 4절 '장애 주입 테스트', M7). LLM·MCP 서버 없이 실행한다.

도구가 실패(예외), 멈춤(타임아웃), 빈 결과, 잘못된 응답을 낼 때
Agent가 재시도하거나, 멈추지 않고 사용자에게 알리는지(경고·실패 목록) 확인한다.

가짜 도구를 실제 Toolbox에 넣어 Toolbox.call의 오류 변환·PII 마스킹까지 함께 거친다.

xfail(strict=True)는 1차 평가에서 확인한 미구현 동작이다 (round1_report.md, M8 개선 대상).
고치고 나면 XPASS가 실패로 잡히므로 그때 표시를 지운다.
"""

import asyncio
from datetime import date

import pytest

from src.mcp_servers.common import get_employee, public_employee
from src.nodes.context import gather_context
from src.nodes.executor import execute
from src.nodes.resolver import Attendance, _find_candidates, _finalize
from src.state import ActionItem, EventDraft
from src.tools import Toolbox, ToolCallError

MEETING_DATE = "2026-11-02"


# ── 가짜 도구 ─────────────────────────────────────────────


class FakeTool:
    """Toolbox가 부르는 ainvoke(args, config)만 흉내 낸다. behavior(args)가 결과를 돌려주거나 예외를 낸다."""

    def __init__(self, name: str, behavior):
        self.name = name
        self.behavior = behavior
        self.calls = 0

    async def ainvoke(self, args, config=None):
        self.calls += 1
        return await self.behavior(args, self.calls)


def ok(value):
    async def run(args, n):
        return value
    return run


def fail(message="연결이 끊어졌습니다"):
    async def run(args, n):
        raise RuntimeError(message)
    return run


def hang(seconds=30):
    async def run(args, n):
        await asyncio.sleep(seconds)
        return {}
    return run


def flaky(failures: int, value):
    """처음 failures번은 실패하고 그다음부터 성공 (일시 장애)."""
    async def run(args, n):
        if n <= failures:
            raise RuntimeError("일시적 오류 (503)")
        return value
    return run


def garbage():
    """MCP가 JSON이 아닌 텍스트를 돌려준 경우 (프록시 오류 페이지 등)."""
    async def run(args, n):
        return [{"type": "text", "text": "<html>502 Bad Gateway</html>"}]
    return run


def toolbox(**behaviors) -> Toolbox:
    return Toolbox([FakeTool(name, b) for name, b in behaviors.items()])


def config(tb: Toolbox) -> dict:
    return {"configurable": {"toolbox": tb}}


def employee(emp_id: str) -> dict:
    return public_employee(get_employee(emp_id))


def item(**kw) -> ActionItem:
    base = dict(id="A1", title="로그인 캐시 PR 리뷰", assignee_raw="최우진 책임", due_raw="수요일까지",
                evidence="최우진 책임: 로그인 캐시 PR 리뷰는 수요일까지.", confidence=0.9)
    return ActionItem(**{**base, **kw})


async def finalize(tb: Toolbox, it: ActionItem, cands: list[dict] | None = None) -> ActionItem:
    cands = [employee("E009")] if cands is None else cands
    await _finalize(tb, it, cands, False, None, MEETING_DATE, Attendance([], []), None, {}, {})
    return it


HEALTHY = dict(
    resolve_date=ok({"date": "2026-11-04", "ambiguous": False}),
    check_availability=ok({"absences": []}),
    list_tasks=ok({"tasks": []}),
)


# ── Toolbox: 오류 변환, 재시도, 타임아웃 ───────────────────


def test_tool_error_becomes_readable_error_without_pii():
    tb = toolbox(find_employee=fail("조회 실패: 연락처 010-1234-5678"))
    with pytest.raises(ToolCallError) as e:
        asyncio.run(tb.call("find_employee", {"name": "최우진"}))
    assert "find_employee 실패" in str(e.value)
    assert "010-1234-5678" not in str(e.value)  # 오류 메시지도 출력물이라 마스킹


@pytest.mark.xfail(strict=True, reason="M8: Toolbox에 일시 장애 재시도 없음")
def test_transient_failure_is_retried():
    tool = FakeTool("find_employee", flaky(1, {"matches": []}))
    tb = Toolbox([tool])
    assert asyncio.run(tb.call("find_employee", {"name": "최우진"})) == {"matches": []}
    assert tool.calls == 2


@pytest.mark.xfail(strict=True, reason="M8: Toolbox에 도구 호출 타임아웃 없음 (멈춘 도구를 계속 기다림)")
def test_hanging_tool_times_out_with_tool_error():
    tb = toolbox(check_availability=hang(30))

    async def run():
        # 테스트가 멈추지 않도록 바깥에서 5초 제한. Toolbox가 그 전에 ToolCallError를 내야 통과
        with pytest.raises(ToolCallError):
            await asyncio.wait_for(tb.call("check_availability", {"user_id": "E009"}), timeout=5)

    asyncio.run(run())


# ── Executor: 등록 실패를 알리고 나머지는 계속 ──────────────


def test_executor_reports_failed_task_and_continues():
    created = iter([RuntimeError("DB 잠김"), {"id": "T-036"}])

    async def create_task(args, n):
        result = next(created)
        if isinstance(result, Exception):
            raise result
        return result

    tb = toolbox(create_task=create_task)
    state = {
        "review_status": "approved",
        "action_items": [item(id="A1", assignee_id="E009"),
                         item(id="A2", title="리허설 계획서 공유", assignee_id="E022")],
    }
    out = asyncio.run(execute(state, config(tb)))
    assert out["registered"] == ["T-036 '리허설 계획서 공유' 등록"]
    assert len(out["failed"]) == 1 and out["failed"][0].startswith("[A1] 로그인 캐시 PR 리뷰: create_task 실패")


def test_executor_reports_failed_event():
    tb = toolbox(create_event=fail("캘린더 서버 응답 없음"))
    event = EventDraft(title="운영 이전 리허설", date_raw="11/11(수)", evidence="리허설 11/11 14시",
                       start="2026-11-11T14:00:00+09:00", end="2026-11-11T15:00:00+09:00", attendee_ids=["E022"])
    out = asyncio.run(execute({"review_status": "approved", "event": event, "action_items": []}, config(tb)))
    assert out["registered"] == []
    assert out["failed"][0].startswith("일정 '운영 이전 리허설': create_event 실패")


# ── Resolver: 확인 도구가 실패하거나 빈 결과면 경고로 알림 ──────


def test_availability_and_task_lookup_failures_become_warnings():
    tb = toolbox(resolve_date=HEALTHY["resolve_date"], check_availability=fail(), list_tasks=fail())
    it = asyncio.run(finalize(tb, item()))
    assert it.assignee_id == "E009" and it.due_date == date(2026, 11, 4)
    assert any(w.startswith("가용성 확인 실패") for w in it.warnings)
    assert any(w.startswith("기존 업무 확인 실패") for w in it.warnings)


def test_empty_date_result_asks_user():
    tb = toolbox(**{**HEALTHY, "resolve_date": ok({"date": None, "ambiguous": True})})
    it = asyncio.run(finalize(tb, item()))
    assert it.due_date is None
    assert any("날짜로 바꾸지 못함" in w for w in it.warnings)


def test_unknown_assignee_asks_user():
    tb = toolbox(find_employee=ok({"matches": []}), get_team_members=ok({"team": None, "members": []}), **HEALTHY)
    it = item(assignee_raw="고길동 책임")
    cands, team_level = asyncio.run(_find_candidates(tb, it, {}))
    assert (cands, team_level) == ([], False)
    asyncio.run(finalize(tb, it, cands))
    assert it.assignee_id is None
    assert any("조직도에서 찾지 못함" in w for w in it.warnings)


@pytest.mark.xfail(strict=True, reason="M8: resolve_date 실패를 잡지 않아 Resolver 전체가 멈춤")
def test_date_tool_failure_becomes_warning():
    tb = toolbox(**{**HEALTHY, "resolve_date": fail()})
    it = asyncio.run(finalize(tb, item()))
    assert it.due_date is None
    assert any("기한" in w for w in it.warnings)


@pytest.mark.xfail(strict=True, reason="M8: resolve_date가 JSON이 아닌 응답을 주면 str.get 오류로 멈춤")
def test_malformed_date_response_becomes_warning():
    tb = toolbox(**{**HEALTHY, "resolve_date": garbage()})
    it = asyncio.run(finalize(tb, item()))
    assert it.due_date is None
    assert any("기한" in w for w in it.warnings)


@pytest.mark.xfail(strict=True, reason="M8: 담당자 후보 조회(find_employee) 실패를 잡지 않아 Resolver 전체가 멈춤")
def test_directory_failure_does_not_stop_resolver():
    tb = toolbox(find_employee=fail(), get_team_members=fail())
    cands, team_level = asyncio.run(_find_candidates(tb, item(), {}))
    assert (cands, team_level) == ([], False)  # 후보 없음 → _finalize가 '담당자 지정 필요' 경고


# ── Context: 과거 회의 검색·중복 확인 실패 ─────────────────


@pytest.mark.xfail(strict=True, reason="M8: 과거 회의 검색·중복 확인 실패를 잡지 않아 Context 단계 전체가 멈춤")
def test_context_search_failure_continues_without_context():
    tb = toolbox(search_past_meetings=fail("Chroma 인덱스를 열 수 없음"), find_duplicate_task=fail())
    state = {"meeting_meta": {"date": MEETING_DATE}, "action_items": [item()], "decisions": []}
    out = asyncio.run(gather_context(state, config(tb)))
    assert out["related_context"] == []
    assert out["action_items"][0].duplicate_of is None
