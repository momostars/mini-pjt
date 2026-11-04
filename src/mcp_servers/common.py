"""MCP 서버 공통: 경로, 조직도, SQLite 적재·초기화."""

import json
import os
import sqlite3
from contextlib import contextmanager
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SEED_DIR = DATA_DIR / "seed"
ORG_PATH = DATA_DIR / "org.json"
TASKS_DB = Path(os.getenv("TASKS_DB", str(DATA_DIR / "tasks.db")))
CALENDAR_DB = Path(os.getenv("CALENDAR_DB", str(DATA_DIR / "calendar.db")))

# 응답에 내보내는 직원 필드. 이메일은 개인정보라 내보내지 않는다.
EMPLOYEE_FIELDS = ("id", "name", "team", "title", "position", "role", "status", "leave_until")


@cache
def load_org() -> dict:
    return json.loads(ORG_PATH.read_text(encoding="utf-8"))


def employees() -> list[dict]:
    return load_org()["employees"]


def public_employee(emp: dict) -> dict:
    return {k: emp[k] for k in EMPLOYEE_FIELDS if k in emp}


def get_employee(emp_id: str) -> dict | None:
    return next((e for e in employees() if e["id"] == emp_id), None)


# ── 호칭 ──────────────────────────────────────────────
# 사내에서 부르는 방식이 다양하다: "민준님", "민준 프로", "김 책임", "서연 책임님", "하린씨"
RANK_TITLES = ("수석", "책임", "선임", "사원")
POSITION_TITLES = ("팀장",)
GENERIC_HONORIFICS = ("매니저", "프로", "선배", "님", "씨")  # 직급 정보가 없는 호칭
COMPOUND_SURNAMES = ("황보", "남궁", "선우", "제갈", "독고")


def split_honorific(raw: str) -> tuple[str, str | None]:
    """호칭을 떼어 (이름 부분, 직급·직책)을 돌려준다. '서연 책임님' → ('서연', '책임'), '민준 프로' → ('민준', None)."""
    text = "".join((raw or "").split())
    title = None
    suffixes = sorted(GENERIC_HONORIFICS + RANK_TITLES + POSITION_TITLES, key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if text.endswith(suf) and len(text) > len(suf):
                text = text[: -len(suf)]
                if title is None and suf in RANK_TITLES + POSITION_TITLES:
                    title = suf
                changed = True
                break
    return text, title


def given_names(full: str) -> set[str]:
    """성을 뺀 이름. '김민준' → {'민준'}. 복성처럼 보이면 두 가지로 읽는다: '황보석' → {'보석', '석'}."""
    names = {full[1:]}
    if full[:2] in COMPOUND_SURNAMES and len(full) > 2:
        names.add(full[2:])
    return names


def surnames(full: str) -> set[str]:
    """'황보석' → {'황', '황보'}."""
    names = {full[:1]}
    if full[:2] in COMPOUND_SURNAMES and len(full) > 2:
        names.add(full[:2])
    return names


def load_seed(name: str) -> dict:
    return json.loads((SEED_DIR / name).read_text(encoding="utf-8"))


@contextmanager
def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── 업무 DB ──────────────────────────────────────────────

TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    assignee_id TEXT NOT NULL,
    due_date TEXT,
    status TEXT NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL
)
"""


def reset_tasks(path: Path = TASKS_DB) -> int:
    """업무 DB를 data/seed/tasks.json 상태로 되돌린다. 평가 러너가 케이스마다 호출."""
    seed = load_seed("tasks.json")["tasks"]
    with connect(path) as conn:
        conn.execute(TASKS_SCHEMA)
        conn.execute("DELETE FROM tasks")
        conn.executemany(
            "INSERT INTO tasks VALUES (:id, :title, :assignee_id, :due_date, :status, :source, :created_at)",
            seed,
        )
    return len(seed)


def ensure_tasks(path: Path = TASKS_DB) -> None:
    """DB가 없으면 시드로 만든다. 이미 있으면 그대로 둔다(등록 결과 유지)."""
    with connect(path) as conn:
        conn.execute(TASKS_SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    if empty:
        reset_tasks(path)


# ── 일정 DB ──────────────────────────────────────────────

CALENDAR_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    attendees TEXT NOT NULL,   -- JSON 배열
    start TEXT NOT NULL,       -- ISO 8601 (+09:00)
    end TEXT NOT NULL
)
"""


def reset_calendar(path: Path = CALENDAR_DB) -> int:
    """일정 DB를 data/seed/calendar.json 상태로 되돌린다. 평가 러너가 케이스마다 호출."""
    seed = load_seed("calendar.json")["events"]
    with connect(path) as conn:
        conn.execute(CALENDAR_SCHEMA)
        conn.execute("DELETE FROM events")
        conn.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            [
                (e["id"], e["title"], e["type"], json.dumps(e["attendees"]), e["start"], e["end"])
                for e in seed
            ],
        )
    return len(seed)


def ensure_calendar(path: Path = CALENDAR_DB) -> None:
    with connect(path) as conn:
        conn.execute(CALENDAR_SCHEMA)
        empty = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    if empty:
        reset_calendar(path)


def reset_all() -> None:
    reset_tasks()
    reset_calendar()
