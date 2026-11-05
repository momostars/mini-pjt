"""실행 전 점검: 설정과 외부 의존(LLM, 임베딩, 검색 인덱스, MCP 서버 3개)을 한 번씩 호출해 본다.

    python -m src.healthcheck

긴 평가 중간에 권한이 빠지거나 일일 토큰 한도에 걸리면, Agent는 장애 대응으로 멈추지 않고
"검색 실패" 같은 안내만 붙여 끝까지 진행한다. 그러면 오염된 평가 수치가 정상처럼 보이므로
평가·데모 전에 이 점검을 먼저 돌린다 (M9). 하나라도 실패하면 종료 코드 1.

키 값은 출력하지 않는다. IAM 사용자 이름과 리전만 보여 준다.
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

HINTS = {
    "AccessDenied": "키의 IAM 정책에 이 모델의 bedrock:InvokeModel 권한이 없거나 리전이 다릅니다. "
                    ".env의 AWS_REGION과 키를 확인하세요",
    "Too many tokens per day": "일일 토큰 한도를 다 썼습니다. 한도가 풀릴 때까지 기다리거나 다른 키를 쓰세요",
    "Throttling": "호출이 너무 잦습니다. 잠시 뒤 다시 시도하세요",
    "UnrecognizedClient": "키가 잘못됐거나 만료됐습니다. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY를 확인하세요",
    "InvalidSignature": "비밀 키가 맞지 않습니다. AWS_SECRET_ACCESS_KEY를 확인하세요",
    "ExpiredToken": "임시 자격 증명이 만료됐습니다. 새 키를 받으세요",
    "NoCredentials": ".env에 AWS 키가 없습니다",
}


def _short(e: Exception) -> str:
    text = str(e)
    if "because" in text:  # AccessDenied: 리소스 ARN 대신 이유 부분만
        text = text[text.find("because"):]
    return f"{type(e).__name__}: {text[-160:]}"


def _hint(e: Exception) -> str | None:
    text = f"{type(e).__name__} {e}"
    return next((h for key, h in HINTS.items() if key in text), None)


class Report:
    def __init__(self):
        self.failed = 0

    def check(self, name: str, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            print(f"  ✅ {name:<14} {detail}  ({time.perf_counter() - t0:.1f}s)")
            return True
        except Exception as e:
            self.failed += 1
            print(f"  ❌ {name:<14} {_short(e)}")
            if hint := _hint(e):
                print(f"     → {hint}")
            return False


def _identity() -> str:
    import boto3

    arn = boto3.client("sts", region_name=os.getenv("AWS_REGION")).get_caller_identity()["Arn"]
    return arn.split(":")[-1]  # 계정 번호는 빼고 사용자 경로만


def _llm() -> str:
    from src.llm import get_llm

    out = get_llm(max_tokens=5).invoke("ping").content
    return f"응답 '{str(out)[:10]}'"


def _embedding() -> str:
    from src.llm import get_embeddings

    return f"{len(get_embeddings().embed_query('점검'))}차원"


def _index() -> str:
    from src.retriever import get_vectorstore, search_past_meetings

    count = get_vectorstore()._collection.count()
    if not count:
        raise RuntimeError("인덱스가 비어 있습니다. python -m src.retriever --ingest 를 먼저 실행하세요")
    hits = search_past_meetings("대시보드 위젯 최대 개수", k=1, before="2026-11-02")
    return f"청크 {count}개, 검색 1위 {hits[0].metadata['chunk_id']}"


async def _mcp() -> str:
    from src.tools import open_toolbox

    async with open_toolbox() as tb:
        emp = await tb.call("find_employee", {"name": "황보석"})
        dup = await tb.call("find_duplicate_task", {"title": "운영 이전 리허설 계획 수립"})
        cal = await tb.call("check_availability", {"user_id": "E006", "date": "2026-11-10"})
    return (f"directory {len(emp.get('matches', []))}명 · task 후보 {len(dup.get('candidates', []))}건 · "
            f"calendar 부재 {len(cal.get('absences', []))}건")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.llm import EMBEDDING_MODEL, MODEL, REGION

    print("MeetingOps 실행 전 점검")
    print(f"  리전 {REGION} · LLM {MODEL} · 임베딩 {EMBEDDING_MODEL}")
    report = Report()
    report.check("AWS 자격 증명", _identity)
    report.check("LLM", _llm)
    report.check("임베딩", _embedding)
    report.check("검색 인덱스", _index)
    report.check("MCP 서버 3개", lambda: asyncio.run(_mcp()))

    if report.failed:
        print(f"\n점검 실패 {report.failed}건. 이 상태로 평가·데모를 돌리면 결과를 믿을 수 없습니다.")
        return 1
    print("\n모두 정상입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
