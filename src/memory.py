"""장기 기억: LangGraph Store에 사용자·팀별로 저장한다 (README 4.6절).

| 이름공간 | 키 | 값 | 쓰는 곳 |
|---|---|---|---|
| ("aliases", user_id) | 호칭 원문(공백 제거) | {employee_id, label, count, updated_at} | HITL 담당자 수정에서 학습 → Resolver가 근거로 사용 |
| ("prefs", user_id) | "settings" | {default_due_days} | Resolver: 기한이 없을 때 제안할 기본 기간 |
| ("team", team_id) | "glossary" | {약어: 뜻} | Extractor 프롬프트 참고 자료 |

학습한 값은 판단을 돕는 근거로만 쓰고, 그 값으로 정한 항목에는 경고를 붙여 HITL에서 확인받는다.

    python -m src.memory list
    python -m src.memory set-pref default_due_days 7
    python -m src.memory set-term "SOP2" "스마트오피스 포털 v2"
    python -m src.memory forget-alias "QA 쪽"
"""

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

DEFAULT_USER = "default"
DEFAULT_TEAM = "smartoffice-v2"


def user_id(config: RunnableConfig | None) -> str:
    return ((config or {}).get("configurable") or {}).get("user_id") or DEFAULT_USER


def team_id(config: RunnableConfig | None) -> str:
    return ((config or {}).get("configurable") or {}).get("team_id") or DEFAULT_TEAM


def alias_key(raw: str) -> str:
    return "".join((raw or "").split())


# ── 호칭 매핑 ──────────────────────────────────────────


async def get_alias(store: BaseStore | None, config, raw: str) -> dict | None:
    if store is None or not raw.strip():
        return None
    item = await store.aget(("aliases", user_id(config)), alias_key(raw))
    return item.value if item else None


async def learn_alias(store: BaseStore | None, config, raw: str, employee_id: str, label: str) -> None:
    if store is None or not raw.strip():
        return
    ns, key = ("aliases", user_id(config)), alias_key(raw)
    old = await store.aget(ns, key)
    count = old.value.get("count", 0) + 1 if old and old.value.get("employee_id") == employee_id else 1
    await store.aput(ns, key, {
        "raw": raw, "employee_id": employee_id, "label": label, "count": count,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })


# ── 설정·팀 정보 ──────────────────────────────────────────


async def get_prefs(store: BaseStore | None, config) -> dict:
    if store is None:
        return {}
    item = await store.aget(("prefs", user_id(config)), "settings")
    return item.value if item else {}


async def get_glossary(store: BaseStore | None, config) -> dict:
    if store is None:
        return {}
    item = await store.aget(("team", team_id(config)), "glossary")
    return item.value if item else {}


# ── 저장소 ──────────────────────────────────────────


@asynccontextmanager
async def sqlite_store(path: str | None = None):
    """실행이 끝나도 남는 기억. 경로는 MEMORY_DB (기본 ./.state/memory.sqlite)."""
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    path = path or os.getenv("MEMORY_DB", "./.state/memory.sqlite")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteStore.from_conn_string(path) as store:
        await store.setup()
        yield store


# ── 관리 명령 ──────────────────────────────────────────


async def _cli(args) -> None:
    cfg = {"configurable": {"user_id": args.user, "team_id": args.team}}
    async with sqlite_store() as store:
        if args.cmd == "list":
            aliases = await store.asearch(("aliases", args.user), limit=200)
            print(f"# 호칭 매핑 ({args.user})")
            for it in aliases:
                v = it.value
                print(f"- '{v['raw']}' → {v['label']} ({v['employee_id']}) 수정 {v['count']}회, {v['updated_at']}")
            print(f"\n# 설정\n{await get_prefs(store, cfg) or '(없음)'}")
            print(f"\n# 팀 용어집 ({args.team})\n{await get_glossary(store, cfg) or '(없음)'}")
        elif args.cmd == "set-pref":
            prefs = await get_prefs(store, cfg)
            value = int(args.value) if args.value.isdigit() else (None if args.value in ("", "none") else args.value)
            prefs[args.key] = value
            await store.aput(("prefs", args.user), "settings", prefs)
            print(f"설정 저장: {prefs}")
        elif args.cmd == "set-term":
            glossary = await get_glossary(store, cfg)
            glossary[args.key] = args.value
            await store.aput(("team", args.team), "glossary", glossary)
            print(f"용어 저장: {args.key} = {args.value}")
        elif args.cmd == "forget-alias":
            await store.adelete(("aliases", args.user), alias_key(args.key))
            print(f"호칭 '{args.key}' 매핑 삭제")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="장기 기억 관리")
    parser.add_argument("cmd", choices=["list", "set-pref", "set-term", "forget-alias"])
    parser.add_argument("key", nargs="?")
    parser.add_argument("value", nargs="?", default="")
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--team", default=DEFAULT_TEAM)
    asyncio.run(_cli(parser.parse_args()))


if __name__ == "__main__":
    main()
