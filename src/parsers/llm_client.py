"""
LLM Parser 共用 client 模組。

統一從 ~/.hermes/auth.json 讀取 Trend Micro endpoint 的憑證，
讓所有 LLM parser 使用同一個初始化方式。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import openai

logger = logging.getLogger(__name__)

# 預設模型（可由環境變數覆蓋）
DEFAULT_MODEL = os.environ.get("PARSER_LLM_MODEL", "gpt-4.1-mini")


def make_client() -> tuple[openai.OpenAI, str]:
    """
    建立 OpenAI-compatible client，優先使用 Trend Micro endpoint。
    回傳 (client, model_name)。
    """
    auth_path = Path.home() / ".hermes" / "auth.json"
    if auth_path.exists():
        try:
            with open(auth_path) as f:
                auth = json.load(f)
            pool = auth.get("credential_pool", {})
            creds = pool.get("custom:api.rdsec.trendmicro.com", [])
            cred  = next((c for c in creds if c.get("priority") == 1), None)
            if cred:
                logger.debug("Using Trend Micro endpoint")
                return (
                    openai.OpenAI(
                        base_url=cred["base_url"],
                        api_key=cred["access_token"],
                    ),
                    DEFAULT_MODEL,
                )
        except Exception as e:
            logger.warning(f"Failed to load Trend cred: {e}, falling back to env")

    # fallback：標準 OPENAI_API_KEY
    return (
        openai.OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        ),
        os.environ.get("OPENAI_MODEL", "gpt-4o"),
    )


def extract_json(text: str) -> dict | None:
    """從 LLM 回應文字中抽取第一個 JSON 物件。"""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None
