"""Streamlit 데모 화면 (README 7.3절).

    streamlit run app.py

CLI(src/cli.py)와 같은 그래프를 같은 프로세스에서 호출한다. 버튼을 누를 때마다
체크포인트·기억 저장소·MCP 도구를 열어 그래프를 실행(또는 Command(resume=...)로 재개)하고 닫는다.
실행 상태는 CHECKPOINT_DB(SQLite)에 남으므로 화면이 다시 그려져도 thread_id로 이어 간다.
입력의 정규식 PII는 그래프에 넣기 전에 지우고, 화면에 보이는 회의록 미리보기도 마스킹한다.
"""

import asyncio
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.types import Command

from src.agent import build_graph, run_config, sqlite_checkpointer
from src.guardrails import mask_pii, scrub
from src.mcp_servers.common import get_employee, reset_all
from src.memory import sqlite_store
from src.tools import open_toolbox

SAMPLES_DIR = Path("data/samples")
SAMPLE_LABELS = {
    "20261102_weekly.md": "11/2 주간 개발회의 · 종합 (동명이인, 중복, 결정 충돌)",
    "20261104_planning.md": "11/4 기획회의 · 상대 날짜, 결정 충돌",
    "20261105_incident_retro.md": "11/5 장애 회고 · 개인정보 마스킹, 휴가 충돌",
    "20261109_weekly.md": "11/9 주간회의 · 프롬프트 주입",
    "20261110_tech_seminar.md": "11/10 기술 세미나 · 할 일 없음",
}

st.set_page_config(page_title="MeetingOps Agent", page_icon="🗂️", layout="wide")


# ── 그래프 실행 ──────────────────────────────────────────


async def _invoke(payload, thread_id: str) -> tuple[dict, dict | None]:
    async with sqlite_checkpointer() as saver, sqlite_store() as store, open_toolbox() as tb:
        graph = build_graph(saver, store)
        result = await graph.ainvoke(payload, run_config(thread_id, tb))
        # 실행이 끝났으면 열려 있는 MCP 연결로 업무·일정을 한 번만 조회해 둔다 (화면을 다시 그릴 때 서버를 또 띄우지 않게)
        snapshot = None if "__interrupt__" in result else await _snapshot(tb, result)
        return result, snapshot


def run_graph(payload) -> None:
    """그래프를 실행하고 결과를 세션에 담는다. 검토 대기면 interrupt 내용을, 끝나면 최종 상태와 조회 결과를 저장한다."""
    try:
        with st.spinner("Agent 실행 중… (MCP 서버 연결과 LLM 호출로 10~40초 걸립니다)"):
            result, snapshot = asyncio.run(_invoke(payload, st.session_state.thread_id))
    except Exception as e:  # 오류 메시지도 출력물이라 PII를 지운다
        st.session_state.error = scrub(f"{type(e).__name__}: {e}")
        return
    st.session_state.error = None
    if "__interrupt__" in result:
        st.session_state.pending = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.pending = None
        st.session_state.final = result
        st.session_state.snapshot = snapshot


def start(request: str) -> None:
    st.session_state.thread_id = uuid.uuid4().hex[:8]
    st.session_state.history = []
    run_graph({"request": mask_pii(request)[0]})


def resume(answer: str, label: str | None = None) -> None:
    st.session_state.history.append(label or answer)
    run_graph(Command(resume=answer))


# ── 업무·일정 조회 (MCP: list_tasks, list_events) ────────

CALENDAR_DAYS = 14  # 기준일이 속한 주의 월요일부터 2주
EVENT_TYPES = {"meeting": "회의", "vacation": "휴가", "leave": "휴직", "training": "교육", "work": "작업"}
WEEKDAYS = "월화수목금토일"


def _dump(obj):
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def _calendar_range(state: dict) -> tuple[date, date]:
    """일정 등록이면 그 날짜, 아니면 회의 날짜(없으면 오늘)가 속한 주의 월요일부터 2주."""
    event = _dump(state.get("event")) or {}
    base = event.get("date") or (state.get("meeting_meta") or {}).get("date")
    day = date.fromisoformat(base[:10]) if base else date.today()
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=CALENDAR_DAYS - 1)


def _people(state: dict) -> set[str]:
    """캘린더에 보여 줄 사람: 이번 할 일 담당자와 일정 참석자 (전 직원 일정을 한꺼번에 보여 주지 않는다)."""
    ids = {a["assignee_id"] for a in map(_dump, state.get("action_items") or [])
           if a.get("assignee_id") and not a.get("excluded")}
    ids |= set((_dump(state.get("event")) or {}).get("attendee_ids") or [])
    return ids


async def _snapshot(tb, state: dict) -> dict:
    """실행이 끝난 직후의 업무 목록과 관련 기간 일정. 실패해도 화면은 계속 보이게 오류만 담는다."""
    start, end = _calendar_range(state)
    try:
        tasks = (await tb.call("list_tasks", {}))["tasks"]
        calendar = await tb.call("list_events", {"start": start.isoformat(), "end": end.isoformat()})
    except Exception as e:  # noqa: BLE001 - 조회 실패가 결과 화면 전체를 막지 않게
        return {"error": scrub(str(e))}
    return {"tasks": tasks, "calendar": calendar, "people": sorted(_people(state))}


def parse_registered(lines: list[str]) -> dict[str, dict]:
    """Executor 처리 결과 문장에서 이번 실행으로 바뀐 업무·일정 ID와 변경 종류를 뽑는다."""
    out = {}
    for line in lines:
        if m := re.match(r"(T-\d+) 기한 변경 (\S+) → (\S+)", line):
            out[m[1]] = {"change": "기한 변경", "detail": f"{m[2]} → {m[3]}"}
        elif m := re.match(r"(T-\d+) 기존 업무와 기한 같음", line):
            out[m[1]] = {"change": "변경 없음", "detail": "기존 업무와 기한 같음"}
        elif m := re.match(r"(T-\d+) '", line):
            out[m[1]] = {"change": "신규", "detail": ""}
        elif m := re.match(r"(C-\d+) 일정", line):
            out[m[1]] = {"change": "신규", "detail": ""}
    return out


def _name(emp_id: str) -> str:
    emp = get_employee(emp_id) or {}
    return f"{emp.get('name', emp_id)} {emp.get('title', '')}".strip()


def task_frame(tasks: list[dict], changed: dict[str, dict], only_changed: bool) -> pd.DataFrame:
    rows = []
    for t in tasks:
        mark = changed.get(t["id"])
        if only_changed and not mark:
            continue
        rows.append({
            "이번 실행": {"신규": "🆕 신규", "기한 변경": "✏️ 기한 변경", "변경 없음": "· 변경 없음"}.get(
                (mark or {}).get("change"), ""),
            "업무": t["id"], "제목": t["title"], "담당": _name(t["assignee_id"]),
            "기한": t["due_date"] or "없음", "변경 내용": (mark or {}).get("detail", ""),
            "상태": t["status"], "출처": t.get("source") or "",
        })
    df = pd.DataFrame(rows)
    if not df.empty and not only_changed:  # 전체 보기에서도 이번 실행 항목을 맨 위로
        df = df.sort_values("이번 실행", key=lambda s: s.eq(""), kind="stable")
    return df


def _when(e: dict) -> str:
    s, t = datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"])
    if s.date() == t.date():
        day = f"{s:%m/%d}({WEEKDAYS[s.weekday()]})"
        return day if (s.hour, s.minute, t.hour, t.minute) == (0, 0, 23, 59) else f"{day} {s:%H:%M}~{t:%H:%M}"
    return f"{s:%m/%d}({WEEKDAYS[s.weekday()]}) ~ {t:%m/%d}({WEEKDAYS[t.weekday()]})"


def event_frame(calendar: dict, people: set[str], changed: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for e in calendar.get("events", []):
        related = sorted(set(e["attendees"]) & people)
        if not related and e["id"] not in changed:
            continue  # 이번 회의와 관계없는 사람의 일정은 보여 주지 않는다
        absent = e["type"] in ("vacation", "leave", "training")
        rows.append({
            "표시": "🆕 이번 등록" if e["id"] in changed else ("⚠ 담당자 부재" if absent else ""),
            "일시": _when(e), "일정": e["title"], "유형": EVENT_TYPES.get(e["type"], e["type"]),
            "관련 인원": ", ".join(_name(i) for i in (related or e["attendees"])),
            "일정 ID": e["id"],
        })
    for h in calendar.get("holidays", []):
        d = date.fromisoformat(h)
        rows.append({"표시": "", "일시": f"{d:%m/%d}({WEEKDAYS[d.weekday()]})", "일정": "공휴일", "유형": "공휴일",
                     "관련 인원": "", "일정 ID": ""})
    return pd.DataFrame(rows)


# ── 화면: 입력 ───────────────────────────────────────────


for key, default in {"thread_id": None, "pending": None, "final": None, "snapshot": None, "error": None, "history": []}.items():
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("입력")
    mode = st.radio("입력 방식", ["샘플 회의록", "직접 입력"], horizontal=True)
    if mode == "샘플 회의록":
        sample = st.selectbox("회의록", list(SAMPLE_LABELS), format_func=lambda f: SAMPLE_LABELS[f])
        path = (SAMPLES_DIR / sample).as_posix()
        request = st.text_input("요청", value=f"{path} 회의록 정리해서 등록해줘")
        with st.expander("회의록 미리보기 (개인정보는 가려서 표시)"):
            st.text(mask_pii((SAMPLES_DIR / sample).read_text(encoding="utf-8"))[0])
    else:
        request = st.text_area(
            "회의록 또는 요청",
            height=260,
            placeholder="예: 11/2 회의 메모: 박지훈 선임이 API 에러 코드 문서를 이번 주 금요일까지 정리하기로 함. 등록해줘",
        )

    if st.button("정리 시작", type="primary", width="stretch", disabled=not request.strip()):
        start(request)

    st.divider()
    st.caption("데모를 처음부터 다시 할 때 업무·일정 DB를 시드 상태로 되돌립니다.")
    if st.button("업무·일정 DB 초기화", width="stretch"):
        reset_all()
        st.toast("업무·일정 DB를 data/seed 상태로 되돌렸습니다.")

st.title("MeetingOps Agent")
st.caption("회의록 → 결정 사항·할 일 추출 → 담당자·기한 확정 → **사람의 승인** → 업무·일정 등록")

if st.session_state.error:
    st.error(f"실행 중 오류가 났습니다. {st.session_state.error}")

if not st.session_state.thread_id:
    st.info("왼쪽에서 회의록을 고르거나 붙여 넣고 **정리 시작**을 누르세요. "
            "등록 전에는 항상 검토 표에서 승인을 받습니다.")
    st.stop()

st.caption(f"실행 ID `{st.session_state.thread_id}`"
           + (" · 응답 기록: " + " → ".join(st.session_state.history) if st.session_state.history else ""))


# ── 화면: 검토 (HITL) ─────────────────────────────────────


def render_review(review: dict) -> None:
    meta = review.get("meeting") or {}
    st.subheader(f"검토 요청 · {meta.get('title') or '회의'} ({meta.get('date') or '날짜 미상'})")
    for note in review.get("notices", []):
        st.info(note)
    for c in review.get("decision_conflicts", []):
        st.warning(f"**이전 결정과 충돌** · {c['decision']}\n\n" + "\n".join(f"- {w}" for w in c["conflicts_with"]))

    items = review.get("items", [])
    if items:
        st.dataframe(
            pd.DataFrame([{
                "확인": "⚠" if r["needs_confirmation"] else "",
                "ID": r["id"], "할 일": r["title"], "담당": r["assignee"], "기한": r["due"], "처리": r["action"],
            } for r in items]),
            hide_index=True, width="stretch",
        )
        st.caption("⚠ 표시 항목은 확인이 필요합니다. 항목을 펼치면 근거 문장과 경고를 볼 수 있습니다.")
        for r in items:
            with st.expander(f"{'⚠ ' if r['needs_confirmation'] else ''}[{r['id']}] {r['title']}"):
                st.markdown(f"> {r['evidence']}")
                for w in r["warnings"]:
                    st.warning(w)
                for n in r.get("notes", []):
                    st.caption(f"· {n}")
    if ev := review.get("event"):
        with st.container(border=True):
            st.markdown(f"**일정** · {ev['title']}  \n{ev['start']} ~ {ev['end']}  \n참석: {', '.join(ev['attendees'])}")
            for w in ev["warnings"]:
                st.warning(w)

    st.divider()
    st.markdown(f"**{review.get('question', '')}**")
    bulk = "approve all" in (review.get("question") or "")
    col1, col2, _ = st.columns([1, 1, 3])
    if col1.button("전부 승인" if bulk else "승인", type="primary", width="stretch"):
        resume("approve all" if bulk else "approve", "승인")
        st.rerun()
    if col2.button("거절", width="stretch"):
        resume("reject", "거절")
        st.rerun()
    with st.form("edit", clear_on_submit=True):
        text = st.text_input("수정 요청", placeholder="예: A8 담당자는 박소율로 / A3는 제외 / A7 기한은 11/13로")
        if st.form_submit_button("수정 요청 보내기") and text.strip():
            resume(text.strip(), f"수정: {text.strip()}")
            st.rerun()


def render_confirm_edit(payload: dict) -> None:
    st.subheader("수정 확인")
    for c in payload.get("changes", []):
        st.markdown(f"- {c}")
    for p in payload.get("problems", []):
        st.warning(p)
    st.markdown(f"**{payload.get('question', '')}**")
    col1, col2, _ = st.columns([1, 1, 3])
    if col1.button("반영", type="primary", width="stretch"):
        resume("yes", "수정 반영")
        st.rerun()
    if col2.button("반영하지 않음", width="stretch"):
        resume("no", "수정 취소")
        st.rerun()


# ── 화면: 결과 ───────────────────────────────────────────


def render_final(state: dict) -> None:
    registered, failed = state.get("registered") or [], state.get("failed") or []
    if registered:
        # "기존 업무와 기한 같음, 변경 없음"도 처리 결과에 들어 있어 건수 대신 항목을 그대로 보여 준다
        st.success("처리 결과\n\n" + "\n".join(f"- {r}" for r in registered))
    elif state.get("review_status") == "rejected":
        st.info("거절해서 아무것도 등록하지 않았습니다.")
    for f in failed:
        st.error(f)

    snap = st.session_state.get("snapshot") or {}
    changed = parse_registered(registered)
    task_ids = [i for i in changed if i.startswith("T-")]
    tab_tasks, tab_cal, tab_report = st.tabs(
        [f"업무 ({len(task_ids)}건 처리)", "캘린더", "요약 보고서"])

    with tab_tasks:
        if snap.get("error"):
            st.warning(f"업무 목록을 불러오지 못했습니다: {snap['error']}")
        else:
            st.caption("업무관리 시스템(task-server)에서 승인 직후 다시 조회한 결과입니다.")
            mine = task_frame(snap.get("tasks", []), changed, only_changed=True)
            if mine.empty:
                st.write("이번 실행에서 등록·변경한 업무가 없습니다.")
            else:
                st.dataframe(mine, hide_index=True, width="stretch")
            with st.expander(f"전체 업무 보기 ({len(snap.get('tasks', []))}건)"):
                st.dataframe(task_frame(snap.get("tasks", []), changed, only_changed=False),
                             hide_index=True, width="stretch")

    with tab_cal:
        if snap.get("error"):
            st.warning(f"일정을 불러오지 못했습니다: {snap['error']}")
        else:
            cal = snap.get("calendar", {})
            start, end = cal.get("range", ["", ""])
            people = set(snap.get("people", []))
            st.caption(f"캘린더(calendar-server)에서 조회한 {start} ~ {end} 일정입니다. "
                       "이번 할 일 담당자·일정 참석자와 관련된 일정만 보여 줍니다.")
            events = event_frame(cal, people, changed)
            if events.empty:
                st.write("이 기간에 관련 일정이 없습니다.")
            else:
                st.dataframe(events, hide_index=True, width="stretch")
            if people:
                st.caption("대상: " + ", ".join(_name(p) for p in sorted(people)))

    with tab_report:
        st.markdown(state.get("report", ""))


pending = st.session_state.pending
if pending and pending.get("type") == "confirm_edit":
    render_confirm_edit(pending)
elif pending:
    render_review(pending)
elif st.session_state.final is not None:
    render_final(st.session_state.final)
