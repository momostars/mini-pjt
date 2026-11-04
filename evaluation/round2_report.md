# 2차 자체 평가 결과 (Day 10 · 개선 후)

> 작성 예정. 1차 평가([round1_report.md](round1_report.md))와 같은 조건으로 실행하고 변화를 비교합니다.

## 1. 실행 정보
- 실행 일시:
- 모델: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- 평가셋: [test_queries.csv](test_queries.csv) 20건 (1차와 동일)
- 추적: `evaluation/results/<실행시각>/trace.jsonl` (집계: `python -m src.tracing <파일>`)

## 2. 1차 이후 개선 내용

| 1차 실패 원인 | 수정 내용 | 수정 파일 |
|---|---|---|
| | | |

## 3. 결과 비교

| category | 건수 | 1차 통과 | 2차 통과 | 변화 |
|---|---|---|---|---|
| positive | 5 | | | |
| negative | 3 | | | |
| edge | 7 | | | |
| guardrail | 5 | | | |
| **전체** | **20** | | | |

완성 기준 ([SERVICE.md](../SERVICE.md) 5절): 전체 85% 이상, guardrail 100% → **충족 / 미충족**

## 4. 기능 지표 비교

| 지표 | 목표 | 1차 | 2차 |
|---|---|---|---|
| 할 일 추출 Recall / Precision | ≥ 0.85 / ≥ 0.85 | | |
| 담당자 매칭 정확도 | ≥ 0.90 | | |
| 기한 변환 정확도 | ≥ 0.90 | | |
| 근거 없는 할 일 생성 | 0건 | | |
| 과거 회의 검색 Hit@4 | ≥ 0.90 | | |
| RAGAS Context Precision / Recall / Faithfulness | ≥ 0.80 / 0.85 / 0.90 | | |
| 회의 1건 처리 시간 | ≤ 3분 | | |

## 5. 남은 실패 케이스와 한계

| id | category | 원인 | 남긴 이유 / 향후 방안 |
|---|---|---|---|
| | | | |
