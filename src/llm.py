"""Bedrock LLM·임베딩 생성 (README 7.5절)."""

import os

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

load_dotenv()

MODEL = os.getenv("MODEL", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")


def get_llm(temperature: float = 0.0, max_tokens: int = 4096) -> ChatBedrockConverse:
    """추출·라우팅·검증은 결과가 일정하도록 temperature=0이 기본."""
    return ChatBedrockConverse(
        model=MODEL,
        region_name=REGION,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(model_id=EMBEDDING_MODEL, region_name=REGION)
