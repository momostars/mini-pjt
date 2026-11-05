#!/usr/bin/env bash
# 점검 → 인덱싱 → 데모 1건 → 평가를 한 번에 실행한다 (README 10장).
#
#   ./run.sh            전체 (인-아웃 20건, 기능 지표, RAGAS, pytest). 30분 안팎, 토큰을 많이 쓴다
#   ./run.sh --quick    점검 · 인덱싱 · 데모 1건 · LLM 없는 테스트만 (2~3분)
#
# Windows는 Git Bash에서 실행한다. 다른 파이썬을 쓰려면 PYTHON=경로 ./run.sh
# 점검이 실패하면 멈춘다. 권한·한도 문제로 오염된 결과를 만들지 않기 위해서다.

set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

step() { printf '\n=== %s ===\n' "$*"; }

step "1. 실행 전 점검 (LLM, 임베딩, 검색 인덱스, MCP 서버)"
if ! "$PY" -m src.healthcheck; then
    # 인덱스가 비어 있어서 실패한 것이면 인덱싱 후 한 번 더 점검한다
    step "인덱스를 만든 뒤 다시 점검"
    "$PY" -m src.retriever --ingest
    "$PY" -m src.healthcheck
fi

step "2. 과거 회의록 인덱싱 (data/past_meetings만)"
"$PY" -m src.retriever --ingest

step "3. 데모 1건: 11/2 주간회의 (DB 초기화 후 자동 승인)"
"$PY" -m src.cli data/samples/20261102_weekly.md --reset --auto approve

if [[ $QUICK -eq 1 ]]; then
    step "4. LLM 없는 테스트"
    "$PY" -m pytest evaluation/tests -q -m "not llm"
else
    step "4-1. 인-아웃 테스트 20건"
    "$PY" -m evaluation.run_test_queries
    step "4-2. 기능 지표 (추출·담당자·기한·Router)"
    "$PY" -m evaluation.run_eval
    step "4-3. 검색·RAGAS"
    "$PY" -m evaluation.run_ragas --ragas
    step "4-4. 전체 테스트"
    "$PY" -m pytest evaluation/tests -q
fi

step "5. 업무·일정 DB를 시드 상태로 되돌림"
"$PY" -c "from src.mcp_servers.common import reset_all; reset_all()"

printf '\n완료. 평가 결과는 evaluation/results/ 에 있습니다. 실행 중 오류가 있었다면 위에 ⚠ 경고가 나옵니다.\n'
