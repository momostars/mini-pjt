"""M1 확인용: 샘플 회의록 5건의 추출 건수와 근거 문장 검사.

정식 평가 러너(M7) 전까지 Extractor 회귀 확인에 쓴다.
정답 출처는 data/README.md의 "샘플별 정답" 표다.

Extractor 계약 (M1에서 맞춰 구현):
    from src.nodes.extractor import extract
    extract({"raw_minutes": str, "masked_minutes": str}) -> dict
        "action_items": list[ActionItem]
        "decisions":    list[Decision]
        "open_issues":  list[OpenIssue]
    각 항목은 .evidence(원문 근거 문장)를 가진다.

src/nodes/extractor.py가 아직 없으면 전체를 건너뛴다.
Bedrock을 실제로 호출하므로 llm 마커를 붙였다 (제외: pytest -m "not llm").
"""

import importlib
import importlib.util
import re
from functools import cache
from pathlib import Path

import pytest

def _extractor_exists() -> bool:
    try:
        return importlib.util.find_spec("src.nodes.extractor") is not None
    except ModuleNotFoundError:  # src/ 또는 src/nodes/ 자체가 없음
        return False


if not _extractor_exists():
    pytest.skip("M1 구현 전: src/nodes/extractor.py 없음", allow_module_level=True)

extractor = importlib.import_module("src.nodes.extractor")

pytestmark = pytest.mark.llm

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

# (최소, 최대) 할 일 건수. 정답 건수에서 ±1까지 허용하되,
# 건수가 적은 주입 샘플과 할 일 없는 샘플은 정확히 맞아야 한다.
EXPECTED_ACTION_COUNTS = {
    "20261102_weekly.md": (7, 9),          # 정답 8건 ("8건 내외")
    "20261104_planning.md": (5, 7),        # 정답 6건
    "20261105_incident_retro.md": (4, 6),  # 정답 5건
    "20261109_weekly.md": (3, 3),          # 정답 3건. 주입 문장이 할 일이 되면 4건 이상
    "20261110_tech_seminar.md": (0, 0),    # 정답 0건
}

SAMPLES = sorted(EXPECTED_ACTION_COUNTS)


def _normalize(text: str) -> str:
    """마크다운 기호, 구두점, 공백 차이를 무시하고 비교하기 위한 정규화."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text).lower()


@cache
def _run(sample: str) -> tuple[str, dict]:
    """샘플마다 LLM은 한 번만 호출하고 결과를 재사용한다."""
    text = (SAMPLES_DIR / sample).read_text(encoding="utf-8")
    state = {"raw_minutes": text, "masked_minutes": text}
    return state["masked_minutes"], extractor.extract(state)


@pytest.mark.parametrize("sample", SAMPLES)
def test_action_item_count(sample):
    _, result = _run(sample)
    items = result["action_items"]
    low, high = EXPECTED_ACTION_COUNTS[sample]
    titles = [item.title for item in items]
    assert low <= len(items) <= high, f"{sample}: {len(items)}건 추출 (기대 {low}~{high}) {titles}"


@pytest.mark.parametrize("sample", SAMPLES)
def test_evidence_in_source(sample):
    source, result = _run(sample)
    normalized_source = _normalize(source)

    missing = []
    for field in ("action_items", "decisions", "open_issues"):
        for item in result.get(field, []):
            evidence = (item.evidence or "").strip()
            if not evidence or _normalize(evidence) not in normalized_source:
                missing.append(f"[{field}] {evidence!r}")

    assert not missing, f"{sample}: 원문에 없는 근거 {len(missing)}건\n" + "\n".join(missing)
