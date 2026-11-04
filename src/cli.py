"""실행 진입점 (README 7.3절).

    python -m src.cli data/samples/20261102_weekly.md
    python -m src.cli "11/2 회의 메모: 박지훈 선임이 ... 등록해줘" --auto approve
    python -m src.cli data/samples/20261102_weekly.md --reset     # 업무·일정 DB를 시드로 초기화 후 실행
    python -m src.cli --thread 3f2a9c1b                            # 검토 대기 중인 실행 이어 가기

검토 단계 응답: approve / reject / 'approve all'(20건 이상) / 자연어 수정("A8 담당자는 박소율로")
실행 상태는 CHECKPOINT_DB(SQLite)에 저장된다. 입력의 정규식 PII는 그래프에 넣기 전에 지운다.
"""

import argparse
import asyncio
import sys
import traceback
import uuid

from langgraph.types import Command

from src.agent import build_graph, run_config, sqlite_checkpointer
from src.guardrails import mask_pii
from src.mcp_servers.common import reset_all
from src.tools import open_toolbox


def print_review(review: dict) -> None:
    if review.get("type") == "confirm_edit":
        print("\n=== 수정 확인 ===")
        for c in review.get("changes", []):
            print(f"  - {c}")
        for p in review.get("problems", []):
            print(f"  ⚠ {p}")
        print(f"\n{review.get('question')}")
        return

    meta = review.get("meeting") or {}
    print(f"\n=== 검토 요청: {meta.get('title') or ''} ({meta.get('date') or '날짜 미상'}) ===")
    for n in review.get("notices", []):
        print(f"ℹ {n}")
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


async def _pending_interrupt(graph, config):
    snapshot = await graph.aget_state(config)
    return snapshot.interrupts[0].value if snapshot.interrupts else None


async def run(request: str | None, auto: str | None, thread_id: str) -> dict:
    async with sqlite_checkpointer() as saver, open_toolbox() as tb:
        graph = build_graph(saver)
        config = run_config(thread_id, tb)
        print(f"[thread_id: {thread_id}] (중단되면 --thread {thread_id} 로 이어서 진행)")
        try:
            if request is None:
                pending = await _pending_interrupt(graph, config)
                if pending is None:
                    print("이 thread에는 검토 대기 중인 실행이 없습니다.")
                    return {}
                result = {"__interrupt__": [type("I", (), {"value": pending})()]}
            else:
                result = await graph.ainvoke({"request": mask_pii(request)[0]}, config)
            while "__interrupt__" in result:
                print_review(result["__interrupt__"][0].value)
                if auto:
                    decision = "yes" if result["__interrupt__"][0].value.get("type") == "confirm_edit" else auto
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
    parser.add_argument("request", nargs="?", help="회의록 파일 경로 또는 요청 문장 (생략하면 --thread 이어 가기)")
    parser.add_argument("--auto", choices=["approve", "reject"], help="검토 단계에서 자동 응답")
    parser.add_argument("--reset", action="store_true", help="업무·일정 DB를 data/seed로 초기화")
    parser.add_argument("--thread", default=None, help="thread_id (기본: 새로 발급)")
    args = parser.parse_args()

    if args.request is None and not args.thread:
        parser.error("요청 문장(또는 파일 경로)이나 --thread 중 하나가 필요합니다.")
    if args.reset:
        reset_all()
    asyncio.run(run(args.request, args.auto, args.thread or uuid.uuid4().hex[:8]))


if __name__ == "__main__":
    main()
