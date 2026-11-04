"""콜백 추적: 실행 기록을 trace.jsonl에 남기고 집계한다 (README 4.7절).

외부 추적 서비스(LangSmith 등) 없이 BaseCallbackHandler로 로컬 파일에 한 줄에 이벤트 하나씩 이어 쓴다.

기록 규칙
- 기록 전에 PII를 마스킹한다 (trace 파일도 출력물)
- LLM 프롬프트·응답 본문은 저장하지 않고 토큰 수만 남긴다. 도구 인자·결과는 마스킹 후 500자까지
- 노드 이벤트는 LangGraph 노드만 남기고 노드 안의 내부 체인은 건너뛴다
- 기록 중 오류가 나도 Agent 실행은 멈추지 않는다

    python -m src.tracing logs/trace.jsonl        # 실행별 노드 지연, 토큰, 도구 성공률
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from src.guardrails import scrub

KST = timezone(timedelta(hours=9))
MAX_FIELD_CHARS = 500
INTERRUPT_ERRORS = ("GraphInterrupt", "NodeInterrupt")


def _clip(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = scrub(text)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "…"


def _tool_output(output: Any) -> Any:
    content = getattr(output, "content", output)  # ToolMessage면 본문만
    if isinstance(content, list) and all(isinstance(b, dict) and "text" in b for b in content):
        content = "".join(b["text"] for b in content)  # MCP 텍스트 블록이면 본문만
    return content


class JsonlTraceHandler(BaseCallbackHandler):
    """그래프 실행 1건(trace_id = thread_id)의 노드·LLM·도구 이벤트를 JSONL로 기록한다."""

    raise_error = False  # 핸들러 오류가 실행을 멈추지 않게

    def __init__(self, path: str | Path | None = None, trace_id: str = ""):
        self.path = Path(path or os.getenv("TRACE_PATH", "./logs/trace.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_id = trace_id
        self._starts: dict[UUID, tuple[float, str, str]] = {}  # run_id → (시작 시각, 종류, 이름)
        self._lock = threading.Lock()

    # ── 기록 ──

    def _write(self, event: str, run_id: UUID, parent_run_id: UUID | None, **fields) -> None:
        try:
            row = {
                "ts": datetime.now(KST).isoformat(timespec="milliseconds"),
                "trace_id": self.trace_id,
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "event": event,
                **fields,
            }
            line = json.dumps(row, ensure_ascii=False, default=str)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:  # 기록 실패로 Agent를 멈추지 않는다
            print(f"[trace] 기록 실패: {e}", file=sys.stderr)

    def _start(self, run_id: UUID, kind: str, name: str) -> None:
        self._starts[run_id] = (time.perf_counter(), kind, name)

    def _end(self, run_id: UUID) -> tuple[int | None, str | None, str | None]:
        started = self._starts.pop(run_id, None)
        if not started:
            return None, None, None
        t0, kind, name = started
        return round((time.perf_counter() - t0) * 1000), kind, name

    # ── 노드 ──

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        try:
            node = (metadata or {}).get("langgraph_node")
            name = kwargs.get("name") or (serialized or {}).get("name")
            if node and name == node:  # 노드 안의 내부 체인(프롬프트 | LLM 등)은 건너뜀
                self._start(run_id, "node", node)
                self._write("node_start", run_id, parent_run_id, node=node)
        except Exception:
            pass

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        try:
            if run_id in self._starts and self._starts[run_id][1] == "node":
                latency, _, node = self._end(run_id)
                self._write("node_end", run_id, parent_run_id, node=node, latency_ms=latency)
        except Exception:
            pass

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        try:
            if run_id not in self._starts:
                return
            latency, kind, name = self._end(run_id)
            if type(error).__name__ in INTERRUPT_ERRORS:  # HITL 검토 대기는 오류가 아니다
                self._write("interrupt", run_id, parent_run_id, node=name, latency_ms=latency)
            else:
                self._write("error", run_id, parent_run_id, name=name, error_type=type(error).__name__,
                            message=_clip(str(error)))
        except Exception:
            pass

    # ── LLM ──

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        try:
            model = (metadata or {}).get("ls_model_name") or (serialized or {}).get("kwargs", {}).get("model", "")
            self._start(run_id, "llm", model)
            self._write("llm_start", run_id, parent_run_id, model=model,
                        node=(metadata or {}).get("langgraph_node"))
        except Exception:
            pass

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        try:
            latency, _, model = self._end(run_id)
            usage = {}
            for gen in (response.generations or [[]])[0]:
                msg = getattr(gen, "message", None)
                if msg is not None and getattr(msg, "usage_metadata", None):
                    usage = msg.usage_metadata
                    break
            if not usage and response.llm_output:
                u = response.llm_output.get("usage") or {}
                usage = {"input_tokens": u.get("prompt_tokens") or u.get("inputTokens"),
                         "output_tokens": u.get("completion_tokens") or u.get("outputTokens")}
            self._write("llm_end", run_id, parent_run_id, model=model, latency_ms=latency,
                        input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
        except Exception:
            pass

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        try:
            _, _, model = self._end(run_id)
            self._write("error", run_id, parent_run_id, name=model, error_type=type(error).__name__,
                        message=_clip(str(error)))
        except Exception:
            pass

    # ── 도구 ──

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None,
                      inputs=None, **kwargs):
        try:
            name = kwargs.get("name") or (serialized or {}).get("name", "")
            self._start(run_id, "tool", name)
            self._starts[run_id] = (*self._starts[run_id], inputs if inputs is not None else input_str)
            self._write("tool_start", run_id, parent_run_id, tool=name, args=_clip(inputs if inputs is not None else input_str))
        except Exception:
            pass

    def _pop_tool(self, run_id):
        started = self._starts.pop(run_id, None)
        if not started:
            return None, None, None
        t0, _, name, args = started
        return round((time.perf_counter() - t0) * 1000), name, args

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        try:
            latency, name, args = self._pop_tool(run_id)
            self._write("tool_end", run_id, parent_run_id, tool=name, args=_clip(args), output=_clip(_tool_output(output)),
                        latency_ms=latency)
        except Exception:
            pass

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        try:
            latency, name, args = self._pop_tool(run_id)
            self._write("error", run_id, parent_run_id, name=name, tool=name, args=_clip(args),
                        error_type=type(error).__name__, message=_clip(str(error)), latency_ms=latency)
        except Exception:
            pass


# ── 읽기·집계 ──────────────────────────────────────────


def load_events(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tools_called(events: list[dict], trace_id: str | None = None) -> set[str]:
    """성공한 도구 호출 이름 (평가 러너의 expected_tools 판정용)."""
    return {e["tool"] for e in events if e["event"] == "tool_end" and (trace_id is None or e["trace_id"] == trace_id)}


def summarize(events: list[dict]) -> dict[str, dict]:
    runs: dict[str, dict] = defaultdict(lambda: {
        "nodes": defaultdict(lambda: {"count": 0, "latency_ms": 0}),
        "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "llm_latency_ms": 0,
        "tool_ok": defaultdict(int), "tool_err": defaultdict(int), "interrupts": 0, "errors": 0,
        "start": None, "end": None,
    })
    for e in events:
        r = runs[e["trace_id"]]
        r["start"] = r["start"] or e["ts"]
        r["end"] = e["ts"]
        ev = e["event"]
        if ev == "node_end":
            r["nodes"][e["node"]]["count"] += 1
            r["nodes"][e["node"]]["latency_ms"] += e.get("latency_ms") or 0
        elif ev == "llm_end":
            r["llm_calls"] += 1
            r["input_tokens"] += e.get("input_tokens") or 0
            r["output_tokens"] += e.get("output_tokens") or 0
            r["llm_latency_ms"] += e.get("latency_ms") or 0
        elif ev == "tool_end":
            r["tool_ok"][e["tool"]] += 1
        elif ev == "interrupt":
            r["interrupts"] += 1
        elif ev == "error":
            r["errors"] += 1
            if e.get("tool"):
                r["tool_err"][e["tool"]] += 1
    return runs


def print_summary(runs: dict[str, dict]) -> None:
    for trace_id, r in runs.items():
        ok, err = sum(r["tool_ok"].values()), sum(r["tool_err"].values())
        rate = f"{ok / (ok + err):.0%}" if ok + err else "-"
        print(f"\n## {trace_id}  ({r['start']} ~ {r['end']})")
        print(f"LLM {r['llm_calls']}회, 토큰 입력 {r['input_tokens']:,} / 출력 {r['output_tokens']:,}, "
              f"LLM 지연 합계 {r['llm_latency_ms'] / 1000:.1f}s")
        print(f"도구 호출 {ok + err}회, 성공률 {rate}, 검토 대기 {r['interrupts']}회, 오류 {r['errors']}건")
        print("노드별 지연 (합계 ms / 실행 횟수):")
        for node, s in sorted(r["nodes"].items(), key=lambda kv: -kv[1]["latency_ms"]):
            print(f"  {node:<14} {s['latency_ms']:>7,} ms / {s['count']}회")
        tools = sorted(set(r["tool_ok"]) | set(r["tool_err"]))
        if tools:
            print("도구별 (성공/실패): " + ", ".join(f"{t} {r['tool_ok'][t]}/{r['tool_err'][t]}" for t in tools))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="trace.jsonl 집계")
    parser.add_argument("path", nargs="?", default=os.getenv("TRACE_PATH", "./logs/trace.jsonl"))
    parser.add_argument("--trace", help="이 trace_id만 보기")
    args = parser.parse_args()
    events = load_events(args.path)
    if args.trace:
        events = [e for e in events if e["trace_id"] == args.trace]
    print_summary(summarize(events))


if __name__ == "__main__":
    main()
