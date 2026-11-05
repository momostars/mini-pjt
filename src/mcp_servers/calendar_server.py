"""calendar-server: 일정 등록과 가용성 확인 (README 4.3절).

데이터는 data/calendar.db(SQLite). 처음 실행할 때 data/seed/calendar.json으로 채운다.
"""

import json
from datetime import date, datetime

from mcp.server.fastmcp import FastMCP

from src.mcp_servers.common import CALENDAR_DB, connect, ensure_calendar, get_employee, load_seed

mcp = FastMCP("calendar-server", log_level="WARNING")

ABSENCE_TYPES = ("vacation", "leave", "training")  # 업무를 맡기 어려운 일정


def _event(row) -> dict:
    e = dict(row)
    e["attendees"] = json.loads(e["attendees"])
    return e


@mcp.tool()
def create_event(title: str, start: str, end: str, attendees: list[str]) -> dict:
    """캘린더 일정을 등록한다. 사용자가 일정 등록을 직접 요청하고 HITL에서 승인했을 때만 호출한다.

    start, end: ISO 8601 (예: 2026-11-11T14:00:00+09:00). attendees: 직원 ID 목록.
    """
    s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    if e <= s:
        raise ValueError("종료 시각이 시작 시각보다 늦어야 합니다.")
    unknown = [a for a in attendees if not get_employee(a)]
    if unknown:
        raise ValueError(f"조직도에 없는 직원 ID입니다: {unknown}")

    with connect(CALENDAR_DB) as conn:
        nums = [int(r[0].split("-")[1]) for r in conn.execute("SELECT id FROM events WHERE id LIKE 'C-2%'")]
        event = {
            "id": f"C-{max(nums, default=200) + 1}",
            "title": title,
            "type": "meeting",
            "attendees": attendees,
            "start": s.isoformat(),
            "end": e.isoformat(),
        }
        conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (event["id"], title, "meeting", json.dumps(attendees), event["start"], event["end"]),
        )
    return event


@mcp.tool()
def check_availability(user_id: str, date: str, end_date: str | None = None) -> dict:
    """직원의 특정 날짜(또는 date~end_date 기간) 일정과 부재(휴가·휴직·교육)를 확인한다.

    할 일 기한을 정할 때는 회의 날짜~기한을 기간으로 넣어 휴가와 겹치는지 본다.
    available=False면 기간 중 부재가 있으므로 HITL에서 확인받는다.
    """
    emp = get_employee(user_id)
    if not emp:
        raise ValueError(f"조직도에 없는 직원 ID입니다: {user_id}")

    start_day = _parse_day(date)
    end_day = _parse_day(end_date) if end_date else start_day
    if end_day < start_day:
        start_day, end_day = end_day, start_day

    with connect(CALENDAR_DB) as conn:
        rows = [_event(r) for r in conn.execute("SELECT * FROM events ORDER BY start")]

    overlapping = [
        e for e in rows
        if user_id in e["attendees"]
        and datetime.fromisoformat(e["start"]).date() <= end_day
        and datetime.fromisoformat(e["end"]).date() >= start_day
    ]
    absences = [e for e in overlapping if e["type"] in ABSENCE_TYPES]
    holidays = [
        h for h in load_seed("calendar.json")["holidays"]
        if start_day <= _parse_day(h) <= end_day
    ]

    return {
        "user_id": user_id,
        "name": emp["name"],
        "status": emp["status"],
        "range": [start_day.isoformat(), end_day.isoformat()],
        "available": emp["status"] != "on_leave" and not absences,
        "absences": absences,
        "events": [e for e in overlapping if e["type"] not in ABSENCE_TYPES],
        "holidays": holidays,
    }


MAX_LIST_DAYS = 31  # 한 번에 조회할 수 있는 기간. 일정 전체를 통째로 내보내지 않는다


@mcp.tool()
def list_events(start: str, end: str, user_id: str | None = None) -> dict:
    """start~end(YYYY-MM-DD, 양 끝 포함) 기간과 겹치는 일정과 공휴일을 조회한다. 조회 전용 (M9 이후 화면용).

    user_id를 주면 그 직원이 참석하는 일정만. 기간은 최대 31일.
    """
    start_day, end_day = _parse_day(start), _parse_day(end)
    if end_day < start_day:
        raise ValueError("종료일이 시작일보다 빠릅니다.")
    if (end_day - start_day).days + 1 > MAX_LIST_DAYS:
        raise ValueError(f"한 번에 {MAX_LIST_DAYS}일까지만 조회할 수 있습니다.")
    if user_id and not get_employee(user_id):
        raise ValueError(f"조직도에 없는 직원 ID입니다: {user_id}")

    with connect(CALENDAR_DB) as conn:
        rows = [_event(r) for r in conn.execute("SELECT * FROM events ORDER BY start, id")]
    events = [
        e for e in rows
        if datetime.fromisoformat(e["start"]).date() <= end_day
        and datetime.fromisoformat(e["end"]).date() >= start_day
        and (user_id is None or user_id in e["attendees"])
    ]
    holidays = [h for h in load_seed("calendar.json")["holidays"] if start_day <= _parse_day(h) <= end_day]
    return {"range": [start_day.isoformat(), end_day.isoformat()], "events": events, "holidays": holidays}


def _parse_day(value: str):
    return date.fromisoformat(value[:10])


if __name__ == "__main__":
    ensure_calendar()
    mcp.run()
