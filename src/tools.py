"""도메인 도구: resolve_date(규칙 기반), search_past_meetings(RAG), MCP 도구 로딩.

resolve_date는 LLM을 쓰지 않는다 (README 4.8절 기한 변환 규칙).
"""

import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
import os
import re
import sys
from datetime import date, datetime, timedelta
from functools import cache
from pathlib import Path

from langchain_core.tools import tool

from src.retriever import search_past_meetings as _search

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))

WEEKDAYS = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}

# 기준 일정 별칭 → 캘린더 마일스톤 제목의 앞부분. 표현이 정확히 일치할 때만 쓴다
# ("운영 이전 리허설"을 "운영 이전"으로 잘못 잡지 않도록)
MILESTONE_ALIASES = {
    "배포동결": "백엔드 배포 동결",
    "동결": "백엔드 배포 동결",
    "백엔드배포동결": "백엔드 배포 동결",
    "운영이전": "운영 환경 클라우드 이전",
    "클라우드이전": "운영 환경 클라우드 이전",
    "운영환경클라우드이전": "운영 환경 클라우드 이전",
    "qa시작": "QA 기간 시작",
    "qa기간": "QA 기간 시작",
    "qa": "QA 기간 시작",
    "보안재점검": "보안 재점검",
    "오픈": "스마트오피스 포털 v2 오픈",
    "서비스오픈": "스마트오피스 포털 v2 오픈",
    "v2오픈": "스마트오피스 포털 v2 오픈",
}


# ── 캘린더 기준 정보 ──────────────────────────────────────


@cache
def _calendar() -> dict:
    return json.loads((DATA_DIR / "seed" / "calendar.json").read_text(encoding="utf-8"))


def _holidays() -> set[date]:
    return {date.fromisoformat(d) for d in _calendar()["holidays"]}


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _holidays()


def previous_business_day(d: date) -> date:
    """d 이전(d 제외)의 가장 가까운 근무일."""
    d -= timedelta(days=1)
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def to_business_day(d: date) -> date:
    """주말·공휴일이면 앞쪽 근무일로 당긴다."""
    return d if is_business_day(d) else previous_business_day(d)


def find_milestone(name: str) -> dict | None:
    key = re.sub(r"\s+", "", name).lower()
    prefix = MILESTONE_ALIASES.get(key)
    if not prefix:
        return None
    return next((m for m in _calendar()["milestones"] if m["title"].startswith(prefix)), None)


# ── resolve_date ──────────────────────────────────────


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _result(base: date, expression: str, value: date | None, rule: str, ambiguous=False, note=None) -> dict:
    if value is not None:
        adjusted = to_business_day(value)
        if adjusted != value:
            note = (note + " / " if note else "") + f"{value.isoformat()}이 휴일이라 앞쪽 근무일로 당김"
        value = adjusted
    return {
        "expression": expression,
        "base_date": base.isoformat(),
        "date": value.isoformat() if value else None,
        "ambiguous": ambiguous or value is None,
        "rule": rule,
        "note": note,
    }


def _reference(name: str, reference_date: str | None) -> tuple[datetime | None, str, dict | None]:
    """기준 일정의 (시각, 설명, 마일스톤). reference_date가 있으면 그것을, 없으면 캘린더 마일스톤에서 찾는다."""
    if reference_date:
        return datetime.fromisoformat(reference_date), "reference_date", None
    milestone = find_milestone(name)
    if milestone:
        return datetime.fromisoformat(milestone["date"]), f"마일스톤: {milestone['title']}", milestone
    return None, "", None


def _with_milestone(result: dict, milestone: dict | None) -> dict:
    result["milestone"] = milestone  # 근거 회의록을 찾을 때 쓴다
    return result


def resolve_date_rule(expression: str, base_date: str, reference_date: str | None = None) -> dict:
    """상대 날짜 표현을 실제 날짜로 바꾼다. base_date는 회의 날짜(YYYY-MM-DD)."""
    base = date.fromisoformat(base_date)
    expr = expression.strip()
    text = re.sub(r"\s+", " ", expr)
    note = "오전까지" if "오전" in text else ("오후까지" if "오후" in text else None)
    text = re.sub(r"(오전|오후|중으로|중에|까지|안에|이내|정도|쯤)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    compact = text.replace(" ", "")

    # 절대 날짜
    if m := re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text):
        return _result(base, expr, date(int(m[1]), int(m[2]), int(m[3])), "절대 날짜", note=note)
    if m := re.search(r"(\d{1,2})\s*(?:/|월\s*)(\d{1,2})일?", text):
        d = date(base.year, int(m[1]), int(m[2]))
        if d < base - timedelta(days=180):
            d = date(base.year + 1, d.month, d.day)
        return _result(base, expr, d, "절대 날짜", note=note)

    # 오늘, 내일, 모레
    for word, days in (("오늘", 0), ("내일", 1), ("모레", 2)):
        if compact.startswith(word):
            return _result(base, expr, base + timedelta(days=days), word, note=note)

    # 이번 주 / 다음 주
    if m := re.fullmatch(r"(이번|다음)주(?:([월화수목금토일])요일)?", compact):
        week = _week_start(base) + timedelta(weeks=0 if m[1] == "이번" else 1)
        if m[2]:
            return _result(base, expr, week + timedelta(days=WEEKDAYS[m[2]]), f"{m[1]} 주 {m[2]}요일", note=note)
        friday = week + timedelta(days=4)
        if m[1] == "이번":
            return _result(base, expr, friday, "이번 주 → 이번 주 금요일", note=note)
        return _result(base, expr, friday, "다음 주 → 다음 주 금요일", ambiguous=True,
                       note="'다음 주까지'는 해석이 갈려 금요일로 두고 확인 요청")

    # 요일만: 기준일 이후 가장 가까운 그 요일
    if m := re.fullmatch(r"([월화수목금토일])요일", compact):
        delta = (WEEKDAYS[m[1]] - base.weekday()) % 7 or 7
        return _result(base, expr, base + timedelta(days=delta), "기준일 이후 가장 가까운 요일", note=note)

    # 다음 달 첫째 주
    if re.fullmatch(r"다음달(?:첫째주|초)", compact):
        first = date(base.year + base.month // 12, base.month % 12 + 1, 1)
        week = _week_start(first) if first.weekday() < 5 else _week_start(first) + timedelta(weeks=1)
        return _result(base, expr, week + timedelta(days=4), "다음 달 첫째 주 → 그 주 금요일", ambiguous=True,
                       note="범위가 넓어 금요일로 두고 확인 요청")

    # 기준 일정 기준: "○○ N주 전", "○○ N일 전", "○○ 전날", "○○ 전"
    if m := re.fullmatch(r"(.+?)(\d+)(주|일)전", compact):
        ref, source, ms = _reference(m[1], reference_date)
        if ref is None:
            return _result(base, expr, None, f"기준 일정 '{m[1]}'을 찾지 못함",
                           note="기준 일정 날짜를 reference_date로 넘기거나 사용자에게 확인")
        days = int(m[2]) * (7 if m[3] == "주" else 1)
        return _with_milestone(
            _result(base, expr, ref.date() - timedelta(days=days), f"{source}에서 {m[2]}{m[3]} 전", note=note), ms
        )

    if m := re.fullmatch(r"(.+?)(전날|전)", compact):
        name = m[1]
        ref, source, ms = _reference(name, reference_date)
        if ref is None:
            return _result(base, expr, None, f"기준 일정 '{name}'을 찾지 못함",
                           note="기준 일정 날짜를 reference_date로 넘기거나 사용자에게 확인")
        # 시각이 있는 일정(예: 14:00 리허설) '전까지'는 같은 날 일정 전까지. 종일·자정 일정은 전 근무일
        if m[2] == "전" and (ref.hour, ref.minute) != (0, 0):
            return _with_milestone(
                _result(base, expr, ref.date(), f"{source} 당일({ref:%H:%M}) 전", note=f"{ref:%H:%M} 이전까지"), ms
            )
        return _with_milestone(
            _result(base, expr, previous_business_day(ref.date()), f"{source} 전 근무일", note=note), ms
        )

    return _result(base, expr, None, "규칙에 없는 표현", note="사용자에게 기한 확인 필요")


@tool
def resolve_date(expression: str, base_date: str, reference_date: str | None = None) -> dict:
    """회의록의 상대 기한 표현을 실제 날짜(YYYY-MM-DD)로 바꾼다. LLM 추측 대신 반드시 이 도구를 쓴다.

    expression: 원문 표현 ("수요일까지", "다음 주 금요일", "QA 시작 전까지", "오픈 2주 전", "배포 동결 전날까지").
      "그 전날까지"처럼 지시어가 있으면 기준 일정 이름을 넣어서 부른다 ("배포 동결 전날까지").
    base_date: 회의 날짜 (YYYY-MM-DD).
    reference_date: 기준 일정이 캘린더 마일스톤에 없을 때 그 일정의 날짜·시각 (예: "2026-11-11T14:00").
    결과의 ambiguous가 true이거나 date가 null이면 HITL에서 사용자 확인을 받는다.
    """
    return resolve_date_rule(expression, base_date, reference_date)


# ── search_past_meetings ──────────────────────────────────


@tool
def search_past_meetings(query: str, before: str | None = None, k: int = 4) -> list[dict]:
    """과거 회의록에서 관련 결정, 기존 할 일, 일정 근거를 찾는다.

    query: 찾을 내용 ("대시보드 위젯 최대 개수", "배포 동결 기간").
    before: 이 날짜(YYYY-MM-DD) 이전 회의만 검색. 보통 지금 회의 날짜를 넣는다.
    결과가 질의와 실제로 관련 있는지는 내용을 읽고 판단한다(관련도 점수만으로 판단하지 않음).
    """
    return [
        {
            "source": d.metadata["source"],
            "date": d.metadata["date"],
            "section": d.metadata["section"],
            "score": d.metadata.get("score"),
            "content": d.page_content,
        }
        for d in _search(query, k=k, before=before)
    ]


# ── MCP 도구 ──────────────────────────────────────────


MCP_SERVERS = {
    "directory": "src.mcp_servers.directory_server",
    "task": "src.mcp_servers.task_server",
    "calendar": "src.mcp_servers.calendar_server",
}


def mcp_connections() -> dict:
    """MCP 서버 3개를 stdio 하위 프로세스로 실행하는 설정 (README 7.4절)."""
    return {
        name: {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", module],
            "cwd": str(ROOT),
            "env": dict(os.environ),  # .env의 DB 경로 등을 하위 프로세스에 전달
        }
        for name, module in MCP_SERVERS.items()
    }


async def aload_mcp_tools() -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return await MultiServerMCPClient(mcp_connections()).get_tools()


def load_mcp_tools() -> list:
    return asyncio.run(aload_mcp_tools())


def local_tools() -> list:
    return [resolve_date, search_past_meetings]


class ToolCallError(Exception):
    pass


def _parse_output(out):
    """MCP 도구 결과(텍스트 블록 목록)를 dict/list로 바꾼다. 로컬 도구 결과는 그대로."""
    if isinstance(out, (dict, list)) and not (
        isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("type") == "text"
    ):
        return out
    if isinstance(out, list):
        # FastMCP는 list 반환값을 항목마다 텍스트 블록 하나로 보낸다 → 블록별로 파싱
        texts = [block.get("text", "") for block in out if isinstance(block, dict)]
        if len(texts) != 1:
            return [_parse_output(t) for t in texts]
        out = texts[0]
    if isinstance(out, str):
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return out
    return out


class Toolbox:
    """노드가 도구를 이름으로 부르는 창구. config를 넘겨 콜백 추적(trace)에 도구 호출이 남게 한다."""

    def __init__(self, tools: list):
        self.tools = {t.name: t for t in tools}

    async def call(self, name: str, args: dict, config=None):
        """도구를 호출한다. 실패하면 ToolCallError(사람이 읽을 메시지)를 낸다."""
        from src.guardrails import scrub  # 도구 미들웨어: 인자와 결과에서 PII 제거

        try:
            out = await self.tools[name].ainvoke(scrub(args), config=config)
        except Exception as e:  # MCP 도구 오류(isError), 입력 검증 오류 등
            raise ToolCallError(scrub(f"{name} 실패: {e}")) from e
        return scrub(_parse_output(out))


@asynccontextmanager
async def open_toolbox():
    """MCP 서버 3개와 연결을 유지한 채 Toolbox를 연다. 호출마다 서버를 새로 띄우지 않는다."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools as _load

    client = MultiServerMCPClient(mcp_connections())
    async with AsyncExitStack() as stack:
        tools = []
        for name in MCP_SERVERS:
            session = await stack.enter_async_context(client.session(name))
            tools += await _load(session, server_name=name, handle_tool_errors=False)
        yield Toolbox(tools + local_tools())
