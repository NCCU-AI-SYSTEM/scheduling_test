"""
P5 Parser：Qwen local（骨架）。

說明：
  實際推論需要在 home_wsl（RTX 3060Ti）上執行 Qwen2.5-1.5B 模型。
  本檔案只提供骨架，確保介面與其他 parser 一致。
  TODO：實作本地推論邏輯。
"""

from __future__ import annotations

import logging

from src.parsers.base import ConditionResult

logger = logging.getLogger(__name__)

# TODO: 設定 Qwen2.5-1.5B 的相關路徑與參數
# MODEL_PATH = "/path/to/Qwen2.5-1.5B"
# DEVICE = "cuda"  # RTX 3060Ti（home_wsl）


def _load_model():
    """
    載入 Qwen2.5-1.5B 模型。

    TODO：實作模型載入邏輯，需要：
    - transformers >= 4.37
    - torch（CUDA 版本，對應 RTX 3060Ti）
    - 模型路徑：home_wsl 上的 Qwen2.5-1.5B 模型目錄

    回傳：
        (tokenizer, model) tuple

    例外：
        NotImplementedError：尚未實作
    """
    raise NotImplementedError(
        "Qwen local 推論尚未實作。"
        "需要在 home_wsl（RTX 3060Ti）上設定 Qwen2.5-1.5B 模型。"
        "請安裝 transformers 與 torch（CUDA），並設定 MODEL_PATH。"
    )


def _run_inference(query: str) -> str:
    """
    使用 Qwen2.5-1.5B 執行推論，回傳 JSON 字串。

    參數：
        query: 使用者輸入的自然語言查詢

    回傳：
        str：JSON 格式的課程搜尋條件

    例外：
        NotImplementedError：尚未實作

    TODO：
    - 設計 prompt template（可參考 llm_fewshot.py 的 few-shot 格式）
    - 設定 max_new_tokens、temperature 等生成參數
    - 解析輸出的 JSON
    """
    raise NotImplementedError(
        "Qwen local 推論尚未實作。"
        "需要在 home_wsl（RTX 3060Ti）上設定 Qwen2.5-1.5B 模型。"
    )


def parse(query: str) -> ConditionResult:
    """
    使用本地 Qwen2.5-1.5B 模型解析查詢字串。

    介面與其他 parser 完全一致，方便替換使用。

    參數：
        query: 使用者輸入的自然語言查詢

    回傳：
        ConditionResult：結構化的搜尋條件

    例外：
        NotImplementedError：本地模型推論尚未實作

    TODO：
    1. 在 home_wsl 上設定 Qwen2.5-1.5B 模型
    2. 實作 _load_model() 與 _run_inference()
    3. 考慮模型快取，避免每次呼叫重複載入
    4. 考慮 prompt 格式（可能需要 Qwen chat template）
    """
    logger.warning(
        "qwen_local.parse() 尚未實作，請在 home_wsl（RTX 3060Ti）上設定模型後再使用。"
    )
    raise NotImplementedError(
        "Qwen local 推論尚未實作。"
        "請參閱 TODO 標記，在 home_wsl 上設定 Qwen2.5-1.5B 後再啟用。"
    )
