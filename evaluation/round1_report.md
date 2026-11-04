# 1차 자체 평가 결과 (Day 9)

> 작성 예정. 아래 양식을 채웁니다.

## 1. 실행 정보
- 실행 일시:
- 모델: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- 평가셋: [test_queries.csv](test_queries.csv) 20건
- 실행 명령: `python -m evaluation.run_test_queries`
- 추적: `evaluation/results/<실행시각>/trace.jsonl` (집계: `python -m src.tracing <파일>`)

## 2. 결과 요약

| category | 건수 | 통과 | 통과율 |
|---|---|---|---|
| positive | 5 | | |
| negative | 3 | | |
| edge | 7 | | |
| guardrail | 5 | | |
| **전체** | **20** | | |

완성 기준 ([SERVICE.md](../SERVICE.md) 5절): 전체 85% 이상, guardrail 100% → **충족 / 미충족**

## 3. 기능 지표

| 지표 | 목표 | 결과 |
|---|---|---|
| 할 일 추출 Recall / Precision | ≥ 0.85 / ≥ 0.85 | |
| 담당자 매칭 정확도 | ≥ 0.90 | |
| 기한 변환 정확도 | ≥ 0.90 | |
| 근거 없는 할 일 생성 | 0건 | |
| 과거 회의 검색 Hit@4 | ≥ 0.90 | |
| RAGAS Context Precision / Recall / Faithfulness | ≥ 0.80 / 0.85 / 0.90 | |
| 회의 1건 처리 시간 | ≤ 3분 | |

## 4. 실패 케이스 분석

| id | category | 기대 | 실제 | 원인 | 개선 방안 |
|---|---|---|---|---|---|
| | | | | | |

## 5. 2차 평가까지 개선 계획
1.
2.
3.
