"""directory-server: 조직도 조회 (README 4.3절).

조직도 대량 조회를 막기 위해 이름이나 팀을 반드시 받고, 이메일은 내보내지 않는다.
"""

import re

from mcp.server.fastmcp import FastMCP

from src.mcp_servers.common import employees, load_org, public_employee

mcp = FastMCP("directory-server", log_level="WARNING")

TITLES = ("수석", "책임", "선임", "사원")
POSITIONS = ("팀장",)
MAX_RESULTS = 5


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _parse_name(raw: str) -> tuple[str, str | None]:
    """'이서연 책임님' → ('이서연', '책임'). 호칭 비교용 원문은 따로 쓴다."""
    text = _norm(raw).removesuffix("님")
    title = None
    for t in TITLES + POSITIONS:
        if text.endswith(t) and len(text) > len(t):
            text, title = text[: -len(t)], t
            break
    return text, title


def _same_name_ids(name: str) -> list[str]:
    return [e["id"] for e in employees() if e["name"] == name]


@mcp.tool()
def find_employee(name: str, title: str | None = None, team: str | None = None) -> dict:
    """이름이나 호칭으로 직원을 찾는다.

    name: 실명("김민준"), 실명+직급("이서연 책임"), 호칭("민준님", "박 선임") 모두 가능.
    title: 직급(수석/책임/선임/사원) 또는 "팀장". 동명이인을 줄일 때 사용.
    team: 팀 이름 일부("플랫폼", "QA"). 동명이인을 줄일 때 사용.
    결과가 여러 명이면 후보를 모두 돌려주므로, 하나로 정하지 말고 사용자 확인을 받는다.
    """
    # "이서연 선임(플랫폼)"처럼 괄호 안 소속은 팀 힌트로 쓴다
    paren = re.search(r"\(([^)]*)\)", name or "")
    if paren:
        team = team or paren.group(1)
        name = name.replace(paren.group(0), "")

    raw = _norm(name)
    if len(raw) < 2:
        return {"query": name, "matches": [], "note": "이름이나 호칭을 2글자 이상 입력해야 합니다."}

    base, parsed_title = _parse_name(name)
    title = title or parsed_title

    matches = []
    for emp in employees():
        aliases = {_norm(a) for a in emp.get("aliases", [])}
        if raw in aliases or emp["name"] == base or (base and base in aliases):
            matches.append(emp)

    if title and len(matches) > 1:
        narrowed = [e for e in matches if title in (e["title"], e["position"])]
        matches = narrowed or matches
    if team and len(matches) > 1:
        narrowed = [e for e in matches if _norm(team).removesuffix("팀") in _norm(e["team"])]
        matches = narrowed or matches

    result = []
    for emp in matches[:MAX_RESULTS]:
        item = public_employee(emp)
        homonyms = [i for i in _same_name_ids(emp["name"]) if i != emp["id"]]
        if homonyms:
            item["same_name_ids"] = homonyms
        result.append(item)

    note = None
    if len(result) > 1:
        note = "후보가 여러 명입니다. 맥락으로 확정할 수 없으면 사용자에게 확인을 요청하세요."
    elif result and result[0]["status"] == "on_leave":
        note = f"휴직 중입니다(~{result[0].get('leave_until')}). 할 일을 배정하면 안 됩니다."
    elif result and result[0].get("same_name_ids"):
        note = "동명이인이 있습니다. 소속과 담당 업무가 맥락과 맞는지 확인하세요."
    return {"query": name, "matches": result, "note": note}


def _match_team(query: str) -> dict | None:
    q = _norm(query).removesuffix("쪽").removesuffix("팀").lower()
    if not q:
        return None
    for team in load_org()["teams"]:
        name = _norm(team["name"]).removesuffix("팀").lower()
        if q in (team["id"].lower(), name) or name.startswith(q):
            return team
    return None


@mcp.tool()
def get_team_members(team: str) -> dict:
    """팀 이름이나 표현("QA팀", "QA 쪽", "인프라팀", "플랫폼")으로 팀원 목록을 돌려준다.

    담당자가 팀 단위로만 적혀 있을 때 후보를 찾는 용도. 휴직자도 status와 함께 포함된다.
    """
    found = _match_team(team)
    if not found:
        return {"query": team, "team": None, "members": [], "note": "해당하는 팀이 없습니다."}
    members = [public_employee(e) for e in employees() if e["team"] == found["name"]]
    return {"query": team, "team": found["name"], "leader_id": found["leader_id"], "members": members}


if __name__ == "__main__":
    mcp.run()
