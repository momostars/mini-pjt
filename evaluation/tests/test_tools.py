"""M3 도구 테스트 (LLM 없음): resolve_date, 조직도 함정, 중복 후보, 가용성, 등록 차단.

정답 출처: data/README.md 샘플별 정답, CLAUDE.md 데이터 함정.
업무·일정 DB는 테스트마다 임시 파일에 시드로 새로 만든다.
"""

import pytest

from src.mcp_servers import calendar_server, task_server
from src.mcp_servers.common import reset_calendar, reset_tasks
from src.mcp_servers.directory_server import find_employee, get_team_members
from src.tools import resolve_date_rule


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    tasks_db, calendar_db = tmp_path / "tasks.db", tmp_path / "calendar.db"
    reset_tasks(tasks_db)
    reset_calendar(calendar_db)
    monkeypatch.setattr(task_server, "TASKS_DB", tasks_db)
    monkeypatch.setattr(calendar_server, "CALENDAR_DB", calendar_db)


# ── resolve_date: data/README.md 샘플 정답 ───────────────────

RESOLVE_CASES = [
    # (회의 날짜, 표현, reference_date, 정답, 모호함)
    ("2026-11-02", "다음 주 수요일", None, "2026-11-11", False),
    ("2026-11-02", "이번 주 금요일까지", None, "2026-11-06", False),
    ("2026-11-02", "수요일까지", None, "2026-11-04", False),
    ("2026-11-02", "다음 주 금요일까지", None, "2026-11-13", False),
    ("2026-11-02", "배포 동결 전날까지", None, "2026-11-12", False),
    ("2026-11-02", "다음 주까지", None, "2026-11-13", True),
    ("2026-11-04", "금요일까지", None, "2026-11-06", False),
    ("2026-11-04", "내일까지", None, "2026-11-05", False),
    ("2026-11-04", "이번 주 안에", None, "2026-11-06", False),
    ("2026-11-04", "목요일 오전까지", None, "2026-11-05", False),
    ("2026-11-04", "QA 시작 전까지", None, "2026-11-20", False),
    ("2026-11-04", "오픈 2주 전까지", None, "2026-11-30", False),
    ("2026-11-05", "다음 주 화요일", None, "2026-11-10", False),
    ("2026-11-05", "다음 주 목요일", None, "2026-11-12", False),
    ("2026-11-05", "오늘 중으로", None, "2026-11-05", False),
    ("2026-11-05", "운영 이전 리허설 전까지", "2026-11-11T14:00", "2026-11-11", False),
    ("2026-11-09", "오늘 중으로", None, "2026-11-09", False),
    ("2026-11-09", "이번 주 금요일까지", None, "2026-11-13", False),
    ("2026-11-09", "내일까지", None, "2026-11-10", False),
    ("2026-11-02", "11/20까지", None, "2026-11-20", False),
]


@pytest.mark.parametrize("base, expr, ref, expected, ambiguous", RESOLVE_CASES)
def test_resolve_date(base, expr, ref, expected, ambiguous):
    r = resolve_date_rule(expr, base, ref)
    assert r["date"] == expected, r
    assert r["ambiguous"] is ambiguous, r


def test_resolve_date_unknown_reference_asks():
    # 마일스톤에 없는 기준 일정은 추측하지 않는다 ("운영 이전"으로 잘못 잡으면 안 됨)
    r = resolve_date_rule("운영 이전 리허설 전까지", "2026-11-05")
    assert r["date"] is None and r["ambiguous"]
    assert r["reference_name"] == "운영 이전 리허설"  # 사용자에게 물을 기준 일정 이름 (띄어쓰기 유지)
    # 지시어만 있으면 기준 일정을 알 수 없다
    r = resolve_date_rule("그 전날까지", "2026-11-02")
    assert r["date"] is None and r["reference_name"] is None


def test_unknown_reference_warning_asks_for_reference_date():
    """기준 일정을 못 찾으면 기한을 추측하지 않고, 그 일정의 날짜를 알려 달라고 경고한다 (M8, 3단계 A안)."""
    import asyncio

    from src.mcp_servers.common import get_employee, public_employee
    from src.nodes.resolver import Attendance, _finalize
    from src.state import ActionItem
    from src.tools import Toolbox, resolve_date

    class Stub:  # 가용성·기존 업무 조회는 이 테스트와 무관
        def __init__(self, name, value):
            self.name, self.value = name, value

        async def ainvoke(self, args, config=None):
            return self.value

    tb = Toolbox([resolve_date, Stub("check_availability", {"absences": []}), Stub("list_tasks", {"tasks": []})])
    item = ActionItem(id="A3", title="스테이징 메일 화이트리스트", assignee_raw="장하린 사원", due_raw="운영 이전 리허설 전까지",
                      evidence="인프라팀에서 운영 이전 리허설 전까지 처리.", confidence=0.9)
    asyncio.run(_finalize(tb, item, [public_employee(get_employee("E025"))], False, None, "2026-11-05",
                          Attendance([], []), None, {}, {}))
    assert item.due_date is None
    warning = next(w for w in item.warnings if "기준 일정" in w)
    assert "'운영 이전 리허설' 날짜" in warning and "A3 기한은" in warning


def test_resolve_date_moves_off_holiday():
    # 2026-10-09(금) 한글날 → 10/8(목)
    assert resolve_date_rule("10/9까지", "2026-10-01")["date"] == "2026-10-08"


# ── directory-server: 동명이인·호칭 함정 ───────────────────


def ids(result):
    return sorted(m["id"] for m in result["matches"])


def test_homonym_kim_minjun():
    r = find_employee("김민준")
    assert ids(r) == ["E006", "E018"]
    assert ids(find_employee("김민준", title="책임")) == ["E006"]
    assert ids(find_employee("김민준 선임")) == ["E018"]
    assert ids(find_employee("민준님")) == ["E006"]


def test_homonym_lee_seoyeon():
    assert ids(find_employee("이서연 책임")) == ["E013"]
    assert ids(find_employee("이서연 선임")) == ["E007"]
    assert ids(find_employee("이서연 선임(플랫폼)")) == ["E007"]
    assert ids(find_employee("이서연", team="프론트엔드")) == ["E013"]


@pytest.mark.parametrize("query, expected", [
    ("민준님", ["E006"]),                      # 조직도 등록 호칭이 우선
    ("민준 프로", ["E006", "E018"]),            # 이름+프로 → 동명이인 둘 다 후보
    ("민준프로", ["E006", "E018"]),
    ("김민준 프로", ["E006", "E018"]),
    ("김민준 책임님", ["E006"]),                 # 직급이 있으면 좁힌다
    ("민준 선임님", ["E018"]),
    ("서연 매니저", ["E007", "E013"]),
    ("하린씨", ["E025"]),
    ("하린 프로", ["E025"]),
    ("보석 프로", ["E022"]),                    # 황보석: 황+보석으로도 읽음
    ("황보 프로", ["E022"]),                    # 황보+석으로도 읽음
    ("박 프로", ["E008", "E019"]),              # 성+프로 → 같은 성 전원
    ("황보석프로님", ["E022"]),
])
def test_honorific_variants(query, expected):
    assert ids(find_employee(query)) == expected


# ── 참석자 명단: 판단을 돕는 근거 ──────────────────────────


def _groups(sample: str) -> tuple[list[list[str]], set[str]]:
    from pathlib import Path

    from src.nodes.common import parse_attendance

    present, absent = parse_attendance(Path("data/samples", sample).read_text(encoding="utf-8"))
    groups = [ids(find_employee(r)) for r in present]
    absent_ids = {g[0] for g in (ids(find_employee(r)) for r in absent) if len(g) == 1}
    return [g for g in groups if g], absent_ids


def test_attendance_narrows_when_clear():
    from src.nodes.common import narrow_by_attendance

    groups, _ = _groups("20261105_incident_retro.md")  # "김민준(플랫폼)", "이서연(플랫폼)"
    assert narrow_by_attendance(["E006", "E018"], groups) == "E006"
    assert narrow_by_attendance(["E007", "E013"], groups) == "E007"


def test_attendance_does_not_narrow_when_both_attended():
    from src.nodes.common import narrow_by_attendance

    groups, _ = _groups("20261102_weekly.md")  # 김민준, 이서연이 두 번씩 (동명이인 둘 다 참석)
    assert narrow_by_attendance(["E006", "E018"], groups) is None
    assert narrow_by_attendance(["E007", "E013"], groups) is None


def test_attendance_absent_list():
    groups, absent = _groups("20261109_weekly.md")  # 불참: 김민준 책임(휴가)
    assert absent == {"E006"}
    assert ["E030"] not in groups  # 백승우는 명단에 없지만(전달받은 일) 담당 후보에서 빼지 않는다


def test_no_attendance_line_in_memo():
    from src.nodes.common import narrow_by_attendance, parse_attendance

    assert parse_attendance("11/2 회의 메모: 민준 프로가 ...") == ([], [])
    assert narrow_by_attendance(["E006", "E018"], []) is None


def test_park_seonim_is_ambiguous():
    r = find_employee("박 선임")
    assert ids(r) == ["E008", "E019"]
    assert r["note"]


def test_on_leave_flagged():
    r = find_employee("임성민")
    assert r["matches"][0]["status"] == "on_leave"
    assert "휴직" in r["note"]


def test_no_email_and_no_bulk_lookup():
    r = find_employee("황보석 책임")
    assert ids(r) == ["E022"]
    assert "email" not in r["matches"][0]
    assert find_employee("")["matches"] == []
    assert find_employee("김")["matches"] == []


@pytest.mark.parametrize("query, team, size", [
    ("QA 쪽", "QA팀", 4),
    ("인프라팀", "인프라운영팀", 5),
    ("플랫폼", "플랫폼개발팀", 7),
])
def test_team_members(query, team, size):
    r = get_team_members(query)
    assert r["team"] == team and len(r["members"]) == size
    assert all("email" not in m for m in r["members"])


# ── task-server: 중복 후보 ──────────────────────────────


@pytest.mark.parametrize("title, assignee, expected", [
    ("리허설 계획서 공유", "E022", "T-012"),
    ("로그인 캐시 운영 코드 반영 마무리", "E006", "T-032"),
    ("알림 API 구현 (기한 연장)", "E007", "T-022"),
    ("다이제스트 템플릿 (8시 반영)", "E027", "T-026"),
])
def test_duplicate_found(title, assignee, expected):
    candidates = task_server.find_duplicate_task(title, assignee)["candidates"]
    assert candidates and candidates[0]["id"] == expected


@pytest.mark.parametrize("title, assignee", [
    ("로그인 캐시 PR 리뷰", "E009"),
    ("대시보드 위젯 12개 확대 및 지연 로딩", "E013"),
    ("알림 읽음 처리 누락 버그 확인", "E007"),
    ("메시지 중복 방지 로직", "E007"),
    ("API 에러 코드 문서 정리", "E008"),
])
def test_duplicate_not_found(title, assignee):
    assert task_server.find_duplicate_task(title, assignee)["candidates"] == []


def test_create_update_and_leave_block():
    task = task_server.create_task("API 에러 코드 문서 정리", "E008", "2026-11-06", "memo")
    assert task["id"] == "T-035" and task["status"] == "todo"
    assert task_server.update_task("T-022", "2026-11-12")["after"] == "2026-11-12"
    assert task_server.create_task("기한 없는 일", "E007")["due_date"] is None
    with pytest.raises(ValueError, match="휴직"):
        task_server.create_task("배치 작업 리팩터링", "E011", "2026-11-20")
    with pytest.raises(ValueError):
        task_server.create_task("없는 사람", "E999")


# ── calendar-server: 휴가·휴직 확인 ─────────────────────────


def test_vacation_conflict_kim_minjun():
    r = calendar_server.check_availability("E006", "2026-11-05", "2026-11-12")
    assert r["available"] is False and r["absences"][0]["title"] == "휴가"
    assert calendar_server.check_availability("E006", "2026-11-02", "2026-11-06")["available"] is True


def test_leave_blocks_availability():
    assert calendar_server.check_availability("E011", "2026-11-20")["available"] is False


def test_create_event():
    e = calendar_server.create_event(
        "운영 이전 리허설", "2026-11-11T14:00:00+09:00", "2026-11-11T16:00:00+09:00", ["E022", "E005"]
    )
    assert e["id"] == "C-201"
    with pytest.raises(ValueError):
        calendar_server.create_event("잘못된 시각", "2026-11-11T14:00:00+09:00", "2026-11-11T13:00:00+09:00", [])


def test_list_events_range_and_user_filter():
    week = calendar_server.list_events("2026-11-09", "2026-11-15")
    ids = [e["id"] for e in week["events"]]
    assert {"C-001", "C-005", "C-101", "C-104", "C-105"} <= set(ids)  # 기간과 겹치면 포함 (11월 한 달 휴직도)
    assert week["range"] == ["2026-11-09", "2026-11-15"]
    mine = calendar_server.list_events("2026-11-09", "2026-11-15", user_id="E006")
    assert [e["id"] for e in mine["events"]] == ["C-101"]  # 김민준 책임 휴가만

    # 승인 후 등록한 일정이 바로 보인다
    calendar_server.create_event("운영 이전 리허설", "2026-11-11T14:00:00+09:00", "2026-11-11T16:00:00+09:00", ["E022"])
    assert "C-201" in [e["id"] for e in calendar_server.list_events("2026-11-11", "2026-11-11")["events"]]


def test_list_events_limits():
    with pytest.raises(ValueError, match="31일"):
        calendar_server.list_events("2026-11-01", "2026-12-31")  # 일정 전체를 한 번에 내보내지 않는다
    with pytest.raises(ValueError):
        calendar_server.list_events("2026-11-15", "2026-11-09")
    with pytest.raises(ValueError):
        calendar_server.list_events("2026-11-09", "2026-11-15", user_id="E999")


# ── 결정 충돌 경고 문구 (M8): 같은 파일의 청크가 여럿이면 결정이 적힌 구역을 출처로 ──


def test_conflict_text_cites_the_chunk_that_holds_the_decision():
    from src.nodes.context import best_hit, conflict_text

    hits = [
        {"source": "20260910_design_review.md", "date": "2026-09-10", "title": "메인 대시보드 디자인 리뷰",
         "section": "논의 내용", "content": "이서연 책임: 위젯이 많아지면 첫 화면 로딩이 느려짐."},
        {"source": "20260910_design_review.md", "date": "2026-09-10", "title": "메인 대시보드 디자인 리뷰",
         "section": "결정 사항", "content": "- 메인 대시보드 위젯은 **사용자당 최대 8개**로 제한."},
        {"source": "20261006_weekly.md", "date": "2026-10-06", "title": "주간 개발회의",
         "section": "결정 사항", "content": "- 메인 대시보드 위젯은 사용자당 최대 8개로 제한."},
    ]
    decision = "메인 대시보드 위젯은 사용자당 최대 8개로 제한"
    hit = best_hit(hits, "20260910_design_review.md", decision)
    assert hit["section"] == "결정 사항"  # 다른 파일의 청크는 고르지 않는다
    assert conflict_text(hit, decision) == (
        "2026-09-10 메인 대시보드 디자인 리뷰(20260910_design_review.md) 결정 사항: " + decision
    )
    assert best_hit(hits, "없는_파일.md", decision) is None  # 검색 결과에 없는 출처는 버린다


# ── 같은 담당자 업무와 중복 재확인 (M8): 후보 선택은 코드, 중복 판단은 LLM ──


def test_same_owner_candidates_pick_similar_open_tasks_only():
    from src.nodes.resolver import same_owner_candidates
    from src.state import ActionItem

    def item(id, title, assignee_id, duplicate_of=None):
        return ActionItem(id=id, title=title, assignee_raw="", evidence="-", confidence=0.9,
                          assignee_id=assignee_id, duplicate_of=duplicate_of)

    t026 = {"id": "T-026", "title": "이메일 다이제스트 템플릿 디자인", "assignee_id": "E027", "due_date": "2026-11-05",
            "status": "todo"}
    other = {"id": "T-099", "title": "접근성 점검", "assignee_id": "E027", "due_date": "2026-10-30", "status": "todo"}
    items = [
        item("A1", "이메일 템플릿 마감 (기한 연장)", "E027"),          # 제목에서 '다이제스트'가 빠져도 후보
        item("A2", "이메일 다이제스트 템플릿 마감", "E027", "T-026"),  # 이미 중복 표시 → 다시 보지 않음
        item("A3", "설정 화면 기획서", None),                           # 담당자 미정 → 볼 업무 목록 없음
    ]
    out = same_owner_candidates(items, [[t026, other], [t026], []])
    assert out == {"A1": [t026]}
