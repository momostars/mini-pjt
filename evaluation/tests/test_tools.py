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
    # 지시어만 있으면 기준 일정을 알 수 없다
    assert resolve_date_rule("그 전날까지", "2026-11-02")["date"] is None


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
