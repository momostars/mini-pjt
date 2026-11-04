"""가드레일 노드: 입력 가드레일(Router 앞), 출력 검증(Resolver 뒤, 검토 전). README 4.5절.

개인정보 원문은 상태에 저장하지 않는다. 체크포인트(SQLite)도 출력물이기 때문이다.
회의록 파일은 이 노드 안에서만 읽고, 마스킹한 본문만 상태에 넣는다.
"""

import re
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from src.guardrails import (
    MAX_INPUT_CHARS,
    detect_bypass,
    detect_injection,
    mask_external_names,
    mask_pii,
    validate_items,
)
from src.mcp_servers.common import employees, given_names, split_honorific

ROOT = Path(__file__).resolve().parents[2]
PATH_RE = re.compile(r"[\w./\\-]+\.(?:md|txt)")

TOO_LONG_MESSAGE = (
    f"입력이 너무 깁니다({MAX_INPUT_CHARS:,}자 초과). 회의록을 안건별로 나눠서 넣어 주세요."
)


def internal_names() -> set[str]:
    """마스킹하지 않을 직원 이름: 실명, 성을 뺀 이름("민준"), 등록 호칭의 이름 부분("서연 책임님" → "서연").

    외부인 판정 때는 후보 이름에서도 호칭을 떼고 비교한다 ("민준프로" → "민준").
    """
    names = set()
    for e in employees():
        names.add(e["name"])
        names.update(n for n in given_names(e["name"]) if len(n) >= 2)
        for alias in e.get("aliases", []):
            base, _ = split_honorific(alias)
            if len(base) >= 2:
                names.add(base)
    return names


def load_minutes(request: str) -> tuple[str, str | None]:
    """요청 안의 회의록 파일을 읽는다. (회의록 본문, 파일명). 프로젝트 폴더 밖 파일은 읽지 않는다."""
    for m in PATH_RE.finditer(request):
        path = (ROOT / m.group(0)).resolve()
        if path.is_file() and ROOT in path.parents:
            return path.read_text(encoding="utf-8"), path.name
    return request, None


async def guard_input(state: dict, config: RunnableConfig) -> dict:
    request = state["request"]
    minutes, filename = load_minutes(request)

    if len(minutes) > MAX_INPUT_CHARS or len(request) > MAX_INPUT_CHARS:
        return {"intent": "out_of_scope", "blocked": True, "report": TOO_LONG_MESSAGE,
                "request": mask_pii(request[:200])[0], "done": ["guard_input"]}

    notes: list[str] = []
    masked_request, req_counts = mask_pii(request)
    masked_minutes, counts = mask_pii(minutes) if filename else (masked_request, req_counts)

    masked_minutes, ext = await mask_external_names(masked_minutes, internal_names(), config)
    if not filename:
        masked_request = masked_minutes
    if ext:
        counts["외부인 이름"] = ext

    if counts:
        detail = ", ".join(f"{k} {v}건" for k, v in counts.items())
        notes.append(f"개인정보 {sum(counts.values())}건({detail})을 마스킹했고 원문은 저장하지 않았습니다.")

    injections = detect_injection(masked_minutes)
    if injections:
        notes.append(
            "회의록 안에서 AI에게 내리는 지시문을 발견했습니다"
            f"(예: '{injections[0]}'). 데이터로만 취급하고 따르지 않았습니다."
        )
    if detect_bypass(masked_request):
        notes.append("확인 절차를 생략해 달라는 요청이 있었지만, 승인 없이는 등록하지 않습니다. 검토 후 승인해 주세요.")

    return {
        "request": masked_request,  # 원문 요청을 마스킹본으로 덮어쓴다
        "masked_minutes": masked_minutes,
        "meeting_meta": {"source": filename},
        "guard_notes": notes,
        "injection_detected": bool(injections),
        "done": ["guard_input"],
    }


def validate(state: dict) -> dict:
    """출력 가드레일: 할 일 형식 검증. 오류는 Supervisor가 재실행 여부 판단에 쓴다."""
    meta = state.get("meeting_meta") or {}
    items, errors = validate_items(
        state.get("action_items", []),
        state.get("masked_minutes", ""),
        meta.get("date"),
        {e["id"]: e for e in employees()},
    )
    return {"action_items": items, "validation_errors": errors, "done": ["validate"]}
