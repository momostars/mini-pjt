"""상태와 Pydantic 스키마 (README 3.3절)."""

from datetime import date
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ── 그래프 상태에 들어가는 항목 ──────────────────────────────


class ActionItem(BaseModel):
    id: str
    title: str
    assignee_raw: str                                    # 원문 표현: "김 책임", "QA 쪽". 언급 없으면 ""
    assignee_id: str | None = None                       # 조직도 매칭 결과 (Resolver)
    assignee_candidates: list[str] = Field(default_factory=list)  # 모호할 때 후보 (Resolver)
    due_raw: str | None = None                           # "다음 주 금요일"
    due_date: date | None = None                         # resolve_date 결과 (Resolver)
    due_ambiguous: bool = False                          # "다음 주까지"처럼 해석이 갈리는 기한
    evidence: str                                        # 근거가 된 원문 문장
    duplicate_of: str | None = None                      # 중복 의심 기존 할 일 ID (Context)
    confidence: float


class Decision(BaseModel):
    id: str
    summary: str                                         # "대시보드 위젯 최대 12개, 지연 로딩 적용"
    topic: str                                           # 안건명. 과거 결정 검색 키
    evidence: str
    conflicts_with: list[str] = Field(default_factory=list)  # 충돌하는 과거 결정 출처 (Context)


class OpenIssue(BaseModel):
    """미결 이슈. 할 일로 등록하지 않고 보고서에만 표시."""

    id: str
    summary: str
    follow_up: str | None = None                         # "11/4 기획회의에서 결정" 같은 후속 처리 원문
    evidence: str


class MeetingState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: Literal["minutes", "schedule", "conflict_check", "out_of_scope", "empty"]
    raw_minutes: str
    masked_minutes: str
    meeting_meta: dict                                   # 날짜, 회의명, 참석자
    plan: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_issues: list[OpenIssue]
    related_context: list[Document]
    review_status: Literal["pending", "approved", "edited", "rejected"]
    registered: list[str]                                # 등록된 task / event ID
    next_agent: str


# ── Extractor 구조화 출력용 (LLM이 채우는 필드만) ─────────────
# id, 매칭·변환 결과는 LLM에게 맡기지 않고 코드와 뒤 단계 Agent가 채운다.


class ExtractedActionItem(BaseModel):
    title: str = Field(description="할 일을 짧은 명사구로. 예: '로그인 캐시 PR 리뷰'")
    assignee_raw: str = Field(
        description="담당자 원문 표현 그대로 ('민준님', '이서연 책임', 'QA 쪽', '인프라팀'). 언급이 없으면 빈 문자열"
    )
    due_raw: str | None = Field(
        description="기한 원문 표현 그대로 ('수요일까지', '다음 주까지', 'QA 시작 전까지'). 날짜로 바꾸지 않음. 없으면 null"
    )
    evidence: str = Field(description="이 할 일의 근거가 된 원문 문장을 한 글자도 바꾸지 않고 그대로 복사")
    confidence: float = Field(ge=0.0, le=1.0, description="이 항목이 실제 할 일이라는 확신도")


class ExtractedDecision(BaseModel):
    summary: str = Field(description="결정 내용 한 문장")
    topic: str = Field(description="안건명. 짧은 명사구 ('대시보드 위젯', '다이제스트 발송 시간')")
    evidence: str = Field(description="결정의 근거가 된 원문 문장을 그대로 복사")


class ExtractedOpenIssue(BaseModel):
    summary: str = Field(description="결론이 나지 않은 안건 한 문장")
    follow_up: str | None = Field(description="후속 처리에 대한 원문 표현 ('수요일 기획회의에서 정하기로'). 없으면 null")
    evidence: str = Field(description="근거가 된 원문 문장을 그대로 복사")


class MeetingMeta(BaseModel):
    date: str | None = Field(description="회의 날짜 YYYY-MM-DD. 원문에 없으면 null")
    title: str | None = Field(description="회의명")
    attendees: list[str] = Field(default_factory=list, description="참석자 이름 (원문 표기 그대로)")


class ExtractionResult(BaseModel):
    """회의록 한 건의 추출 결과."""

    meeting_meta: MeetingMeta
    decisions: list[ExtractedDecision]
    action_items: list[ExtractedActionItem]
    open_issues: list[ExtractedOpenIssue]
