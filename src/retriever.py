"""과거 회의록 RAG: 분할, 임베딩, 인덱싱, 검색 (README 4.2절).

인덱싱 대상은 data/past_meetings/ 뿐이다. data/README.md와 evaluation/은 정답이 있어 넣지 않는다.

    python -m src.retriever --ingest
    python -m src.retriever --query "대시보드 위젯 최대 개수" --before 2026-11-02
"""

import argparse
import os
import re
import threading
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.llm import get_embeddings

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PAST_MEETINGS_DIR = DATA_DIR / "past_meetings"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma")
COLLECTION = "past_meetings"

DEFAULT_K = 4


# ── 분할 ──────────────────────────────────────────────


def _parse_meeting(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """회의록 1건을 메타데이터와 (섹션 이름, 본문) 목록으로 나눈다."""
    lines = path.read_text(encoding="utf-8").splitlines()

    ymd = path.name[:8]
    meta = {
        "source": path.name,
        "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}",
        "date_int": int(ymd),
        "title": next((l[2:].strip() for l in lines if l.startswith("# ")), path.stem),
        "attendees": next(
            (l.split(":", 1)[1].strip() for l in lines if "참석" in l and ":" in l), ""
        ),
    }

    sections: list[tuple[str, str]] = []
    section, sub, body = None, None, []

    def flush():
        text = "\n".join(body).strip()
        if section and text:
            name = f"{section} · {sub}" if sub else section
            sections.append((name, text))

    for line in lines:
        if line.startswith("## "):
            flush()
            section, sub, body = line[3:].strip(), None, []
        elif line.startswith("### ") and section:
            flush()
            sub, body = re.sub(r"^\d+\.\s*", "", line[4:].strip()), []
        elif section:
            body.append(line)
    flush()
    return meta, sections


def _split_table_rows(body: str) -> list[str] | None:
    """마크다운 표만 있는 섹션(Action Items)이면 '머리행 + 행 1개' 조각 목록, 아니면 None.
    표 전체를 한 청크로 두면 서로 다른 할 일이 섞여 임베딩돼, 할 일 하나를 찾는 질의와 잘 맞지 않는다 (M8)."""
    lines = [l for l in body.splitlines() if l.strip()]
    if len(lines) < 3 or not all(l.lstrip().startswith("|") for l in lines):
        return None
    head, sep, rows = lines[0], lines[1], lines[2:]
    if not re.fullmatch(r"\|[\s\-:|]+\|", sep.strip()):
        return None
    return [f"{head}\n{sep}\n{row}" for row in rows]


def load_documents(directory: Path = PAST_MEETINGS_DIR, split_rows: bool = True) -> list[Document]:
    """회의록마다 안건 단위 청크를 만든다. 청크 앞에 날짜·회의명·섹션을 붙여 맥락을 살린다.
    split_rows=True면 표만 있는 섹션(Action Items)은 행마다 청크 하나로 나눈다."""
    if directory.resolve() != PAST_MEETINGS_DIR.resolve():
        raise ValueError(f"인덱싱 대상은 {PAST_MEETINGS_DIR}뿐입니다: {directory}")

    docs = []
    for path in sorted(directory.glob("*.md")):
        meta, sections = _parse_meeting(path)
        i = 0
        for name, body in sections:
            header = f"[{meta['date']} {meta['title']} · {name}]"
            pieces = (_split_table_rows(body) if split_rows else None) or [body]
            for piece in pieces:
                docs.append(
                    Document(
                        page_content=f"{header}\n{piece}",
                        metadata={**meta, "section": name, "chunk_id": f"{path.stem}#{i}"},
                    )
                )
                i += 1
    return docs


# ── 저장소 ──────────────────────────────────────────────


_store_lock = threading.Lock()


@cache
def _create_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # 관련도 = 1 - 코사인 거리
    )


def get_vectorstore() -> Chroma:
    """병렬 검색(여러 스레드)에서 Chroma 클라이언트가 동시에 여러 번 만들어지지 않게 잠근다."""
    with _store_lock:
        return _create_vectorstore()


def ingest() -> int:
    """컬렉션을 비우고 다시 인덱싱한다. 청크 수를 돌려준다."""
    store = get_vectorstore()
    store.reset_collection()
    docs = load_documents()
    store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    return len(docs)


# ── 검색 ──────────────────────────────────────────────


def _date_filter(before: str | None) -> dict | None:
    """before(YYYY-MM-DD)보다 이전 회의만 검색한다. 지금 회의 이후 회의록이 섞이지 않게 함."""
    if not before:
        return None
    return {"date_int": {"$lt": int(before.replace("-", ""))}}


def search_past_meetings(
    query: str,
    k: int = DEFAULT_K,
    before: str | None = None,
    mmr: bool = False,
    fetch_k: int = 20,
) -> list[Document]:
    """과거 회의록 청크를 찾는다. 유사도 검색이면 metadata["score"]에 관련도(0~1)를 넣는다."""
    store = get_vectorstore()
    where = _date_filter(before)

    if mmr:
        return store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=where)

    results = store.similarity_search_with_relevance_scores(query, k=k, filter=where)
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(score, 4)
        docs.append(doc)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="과거 회의록 인덱싱·검색")
    parser.add_argument("--ingest", action="store_true", help="data/past_meetings를 다시 인덱싱")
    parser.add_argument("--query", help="검색어")
    parser.add_argument("--before", help="이 날짜(YYYY-MM-DD) 이전 회의만 검색")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--mmr", action="store_true")
    args = parser.parse_args()

    if args.ingest:
        print(f"{ingest()}개 청크 인덱싱 완료 → {CHROMA_DIR}")
    if args.query:
        for doc in search_past_meetings(args.query, k=args.k, before=args.before, mmr=args.mmr):
            score = doc.metadata.get("score", "-")
            print(f"[{score}] {doc.metadata['chunk_id']}  {doc.page_content.splitlines()[0]}")
    if not (args.ingest or args.query):
        parser.print_help()


if __name__ == "__main__":
    main()
