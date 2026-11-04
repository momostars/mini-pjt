"""실행 진입점 (README 7.3절).

    python -m src.cli data/samples/20261102_weekly.md
    python -m src.cli "11/2 회의 메모: 박지훈 선임이 ... 등록해줘" --auto approve
    python -m src.cli data/samples/20261102_weekly.md --reset   # 업무·일정 DB를 시드로 초기화 후 실행
"""

import argparse
import asyncio
import sys
import traceback
import uuid

from langgraph.types import Command

from src.agent import build_graph, run_config
from src.mcp_servers.common import reset_all
from src.tools import open_toolbox


def print_review(review: dict) -> None:
    meta = review.get("meeting") or {}
    print(f"\n=== 검토 요청: {meta.get('title') or ''} ({meta.get('date') or '날짜 미상'}) ===")
    for c in review.get("decision_conflicts", []):
        print(f"⚠ 결정 충돌: {c['decision']}")
        for w in c["conflicts_with"]:
            print(f"    - {w}")
    for row in review.get("items", []):
        mark = "⚠" if row["needs_confirmation"] else " "
        print(f"{mark} [{row['id']}] {row['title']}")
        print(f"    담당: {row['assignee']} | 기한: {row['due']} | 처리: {row['action']}")
        print(f"    근거: {row['evidence']}")
        for w in row["warnings"]:
            print(f"    - {w}")
        for n in row.get("notes", []):
            print(f"    · {n}")
    if ev := review.get("event"):
        print(f"일정: {ev['title']} {ev['start']} ~ {ev['end']}")
        print(f"    참석: {', '.join(ev['attendees'])}")
        for w in ev["warnings"]:
            print(f"    - {w}")
    print(f"\n{review.get('question')}")


async def run(request: str, auto: str | None, thread_id: str) -> dict:
    graph = build_graph()
    async with open_toolbox() as tb:
        config = run_config(thread_id, tb)
        try:
            result = await graph.ainvoke({"request": request}, config)
            while "__interrupt__" in result:
                review = result["__interrupt__"][0].value
                print_review(review)
                if auto:
                    decision = auto
                    print(f"> {decision} (자동)")
                else:
                    decision = input("> ").strip()
                result = await graph.ainvoke(Command(resume=decision), config)
        except Exception:
            # MCP 연결을 닫을 때 나는 오류에 원래 오류가 가려지지 않게 먼저 출력
            traceback.print_exc()
            raise
    print("\n" + result.get("report", ""))
    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="MeetingOps Agent")
    parser.add_argument("request", help="회의록 파일 경로 또는 요청 문장")
    parser.add_argument("--auto", choices=["approve", "reject"], help="검토 단계에서 자동 응답")
    parser.add_argument("--reset", action="store_true", help="업무·일정 DB를 data/seed로 초기화")
    parser.add_argument("--thread", default=None, help="thread_id (기본: 새로 발급)")
    args = parser.parse_args()

    if args.reset:
        reset_all()
    asyncio.run(run(args.request, args.auto, args.thread or uuid.uuid4().hex[:8]))


if __name__ == "__main__":
    main()
