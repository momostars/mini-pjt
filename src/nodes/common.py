"""노드 공통 유틸."""

import os
import re
from datetime import date, datetime

from langchain_core.runnables import RunnableConfig

from src.tools import Toolbox


def toolbox(config: RunnableConfig) -> Toolbox:
    """그래프 실행 시 config["configurable"]["toolbox"]로 넘긴 Toolbox."""
    tb = (config or {}).get("configurable", {}).get("toolbox")
    if tb is None:
        raise RuntimeError("config['configurable']['toolbox']가 없습니다. open_toolbox()로 열어 넘겨 주세요.")
    return tb


def default_year() -> int:
    """연도 없는 날짜("11/2")에 붙일 연도."""
    return int(os.getenv("DEFAULT_YEAR", datetime.now().year))


def detect_meeting_date(text: str) -> str | None:
    """회의 날짜를 규칙으로 찾는다. '일시' 줄 → 첫 YYYY-MM-DD → 첫 M/D 순서."""
    lines = text.splitlines()
    candidates = [l for l in lines if "일시" in l] + lines
    for line in candidates:
        if m := re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", line):
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    if m := re.search(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])", text):
        try:
            return date(default_year(), int(m[1]), int(m[2])).isoformat()
        except ValueError:
            return None
    return None


ATTENDANCE_RE = re.compile(r"^\s*[-*]?\s*(참석자?|불참자?)\s*[:：]\s*(.+)$")


def parse_attendance(text: str) -> tuple[list[str], list[str]]:
    """회의록의 '참석:'·'불참:' 줄에서 이름 표현을 뽑는다. (참석, 불참). 줄이 없으면 빈 목록."""
    present, absent = [], []
    for line in text.splitlines():
        m = ATTENDANCE_RE.match(line)
        if not m:
            continue
        names = [t.strip() for t in re.split(r"[,，、·/]", m[2]) if t.strip()]
        (absent if m[1].startswith("불참") else present).extend(names)
    return present, absent


def narrow_by_attendance(candidate_ids: list[str], present_groups: list[list[str]]) -> str | None:
    """후보가 여럿일 때 참석자 명단으로 한 명이 분명히 가려지면 그 ID. 아니면 None.

    present_groups: 참석자 표현마다 조직도에서 찾은 ID 목록 ("김민준(플랫폼)" → [E006], "김민준" → [E006, E018]).
    한 명만 '분명히 참석'이고 나머지 후보는 참석 가능성조차 없을 때만 고른다. 참석자 명단은 판단을 돕는 근거일 뿐이다.
    """
    sure = {g[0] for g in present_groups if len(g) == 1}
    possible = {i for g in present_groups for i in g}
    in_sure = [c for c in candidate_ids if c in sure]
    if len(in_sure) != 1:
        return None
    others = [c for c in candidate_ids if c != in_sure[0]]
    return in_sure[0] if all(c not in possible for c in others) else None


def employee_label(emp: dict) -> str:
    """'이서연 선임(플랫폼개발팀)' 형태."""
    return f"{emp['name']} {emp['title']}({emp['team']})"
