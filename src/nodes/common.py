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


def employee_label(emp: dict) -> str:
    """'이서연 선임(플랫폼개발팀)' 형태."""
    return f"{emp['name']} {emp['title']}({emp['team']})"
