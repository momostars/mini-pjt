"""task-server: 업무 관리 (README 4.3절).

데이터는 data/tasks.db(SQLite). 처음 실행할 때 data/seed/tasks.json으로 채운다.
삭제와 완료 처리 도구는 두지 않는다(범위 밖, 프롬프트 주입 대비).
"""

import re
from datetime import date

from mcp.server.fastmcp import FastMCP

from src.mcp_servers.common import TASKS_DB, connect, ensure_tasks, get_employee

mcp = FastMCP("task-server", log_level="WARNING")

OPEN_STATUSES = ("todo", "in_progress")
DUPLICATE_THRESHOLD = 0.55        # 같은 담당자(또는 담당자 미정)
OTHER_ASSIGNEE_THRESHOLD = 0.7    # 다른 담당자: 제목이 훨씬 비슷할 때만
# 중복 비교에서 뺄 흔한 말. 이것만 겹쳐서 중복으로 보이지 않게 한다
STOPWORDS = ("작성", "진행", "완료", "마무리", "정리", "공유", "추가", "적용", "구현", "개발", "확인", "및")


def _row(r) -> dict:
    return dict(r)


def _validate_date(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    return date.fromisoformat(value).isoformat()  # 형식이 틀리면 ValueError


def _check_assignee(assignee_id: str) -> dict:
    emp = get_employee(assignee_id)
    if not emp:
        raise ValueError(f"조직도에 없는 직원 ID입니다: {assignee_id}")
    if emp["status"] == "on_leave":
        raise ValueError(f"{emp['name']}({assignee_id})은 휴직 중이라 업무를 배정할 수 없습니다.")
    return emp


@mcp.tool()
def create_task(title: str, assignee_id: str, due_date: str | None = None, source: str | None = None) -> dict:
    """업무를 새로 등록한다. 사용자가 HITL 검토에서 승인한 항목에만 호출한다.

    due_date: YYYY-MM-DD. 기한이 없으면 비워 둔다(임의로 채우지 않음).
    source: 근거 회의록 파일명.
    """
    _check_assignee(assignee_id)
    due = _validate_date(due_date)
    with connect(TASKS_DB) as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM tasks")]
        next_no = max((int(i.split("-")[1]) for i in ids), default=0) + 1
        task = {
            "id": f"T-{next_no:03d}",
            "title": title,
            "assignee_id": assignee_id,
            "due_date": due,
            "status": "todo",
            "source": source,
            "created_at": date.today().isoformat(),
        }
        conn.execute(
            "INSERT INTO tasks VALUES (:id, :title, :assignee_id, :due_date, :status, :source, :created_at)",
            task,
        )
    return task


@mcp.tool()
def update_task(task_id: str, due_date: str | None) -> dict:
    """기존 업무의 기한만 바꾼다. 사용자가 HITL 검토에서 승인한 항목에만 호출한다."""
    due = _validate_date(due_date)
    with connect(TASKS_DB) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise ValueError(f"없는 업무 ID입니다: {task_id}")
        conn.execute("UPDATE tasks SET due_date = ? WHERE id = ?", (due, task_id))
    return {"id": task_id, "title": row["title"], "before": row["due_date"], "after": due}


@mcp.tool()
def list_tasks(assignee_id: str | None = None, status: str | None = None) -> dict:
    """업무 목록. assignee_id로 담당자의 기존 업무를 확인할 때 쓴다. status: todo/in_progress/done."""
    sql, args = "SELECT * FROM tasks WHERE 1=1", []
    if assignee_id:
        sql += " AND assignee_id = ?"
        args.append(assignee_id)
    if status:
        sql += " AND status = ?"
        args.append(status)
    with connect(TASKS_DB) as conn:
        return {"tasks": [_row(r) for r in conn.execute(sql + " ORDER BY id", args)]}


def _bigrams(text: str) -> set[str]:
    for word in STOPWORDS:
        text = text.replace(word, " ")
    grams = set()
    for token in re.findall(r"[0-9A-Za-z가-힣]+", text.lower()):
        if len(token) == 1:
            grams.add(token)
        grams.update(token[i : i + 2] for i in range(len(token) - 1))
    return grams


def similarity(a: str, b: str) -> float:
    """글자 bigram Dice 계수. 표현이 조금 달라도 같은 일을 잡기 위함."""
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


@mcp.tool()
def find_duplicate_task(title: str, assignee_id: str | None = None) -> dict:
    """진행 중(todo, in_progress)인 업무 중 제목이 비슷한 것을 찾는다.

    assignee_id를 주면 다른 담당자의 업무는 제목이 훨씬 비슷할 때만 후보로 낸다.
    글자 비교로 후보만 돌려주므로("리허설 진행"과 "리허설 계획 수립"도 후보가 됨)
    실제 중복 여부는 맥락으로 판단하고, 중복 의심이면 HITL에서 확인받는다.
    """
    with connect(TASKS_DB) as conn:
        rows = [_row(r) for r in conn.execute(
            f"SELECT * FROM tasks WHERE status IN ({','.join('?' * len(OPEN_STATUSES))})", OPEN_STATUSES
        )]

    candidates = []
    for task in rows:
        score = similarity(title, task["title"])
        same = assignee_id is None or task["assignee_id"] == assignee_id
        threshold = DUPLICATE_THRESHOLD if same else OTHER_ASSIGNEE_THRESHOLD
        if score >= threshold:
            candidates.append({**task, "score": round(score, 3), "same_assignee": same})

    candidates.sort(key=lambda t: t["score"], reverse=True)
    return {"query": title, "candidates": candidates[:3]}


if __name__ == "__main__":
    ensure_tasks()
    mcp.run()
