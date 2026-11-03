"""
Parser 評估指標模組。

提供 slot_f1、exact_match、negation_f1、aggregate_metrics 等函數，
用於評估各 parser 解析結果的品質。
"""

from __future__ import annotations

from typing import Any

from src.parsers.base import ConditionResult

# 所有維度名稱
DIMENSIONS = ["course_lang", "weekday", "hour_range", "point", "kind", "lmt_kind", "unit"]


def _to_set(value: Any) -> set:
    """
    將條件值統一轉換成 set，方便計算 F1。

    - list  → frozenset（可雜湊化）
    - float → {float}
    - None  → 空 set
    - 其他  → {value}

    參數：
        value: 任意條件值

    回傳：
        set：可用於集合運算的值
    """
    if value is None:
        return set()
    if isinstance(value, list):
        return set(tuple(v) if isinstance(v, list) else v for v in value)
    return {value}


def _compute_prf(pred_set: set, gold_set: set) -> dict:
    """
    計算單一維度的 Precision / Recall / F1。

    參數：
        pred_set: 預測值的 set
        gold_set: 黃金標準的 set

    回傳：
        dict：{"p": float, "r": float, "f1": float}
    """
    if not pred_set and not gold_set:
        # 都是空，視為完全正確
        return {"p": 1.0, "r": 1.0, "f1": 1.0}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"p": precision, "r": recall, "f1": f1}


def slot_f1(pred: ConditionResult, gold: ConditionResult) -> dict:
    """
    計算每個維度的 Precision / Recall / F1。
    分別計算 must 和 must_not 的各維度指標，以及整體平均。

    參數：
        pred: 預測的 ConditionResult
        gold: 黃金標準的 ConditionResult

    回傳：
        dict：結構如下
        {
            "must": {
                "course_lang": {"p": float, "r": float, "f1": float},
                ...
            },
            "must_not": {
                "course_lang": {"p": float, "r": float, "f1": float},
                ...
            },
            "overall": {"p": float, "r": float, "f1": float}
        }
    """
    result: dict = {"must": {}, "must_not": {}}
    all_f1_values: list[float] = []

    for slot in ("must", "must_not"):
        pred_dict = getattr(pred, slot)
        gold_dict = getattr(gold, slot)

        for dim in DIMENSIONS:
            pred_val = _to_set(pred_dict.get(dim))
            gold_val = _to_set(gold_dict.get(dim))
            prf = _compute_prf(pred_val, gold_val)
            result[slot][dim] = prf
            all_f1_values.append(prf["f1"])

    # 整體平均 F1
    overall_f1 = sum(all_f1_values) / len(all_f1_values) if all_f1_values else 0.0
    result["overall"] = {"f1": overall_f1}

    return result


def exact_match(pred: ConditionResult, gold: ConditionResult) -> bool:
    """
    檢查 must 和 must_not 是否全部完全正確（完全比對）。

    參數：
        pred: 預測的 ConditionResult
        gold: 黃金標準的 ConditionResult

    回傳：
        bool：must 和 must_not 的所有維度完全一致則為 True
    """
    for slot in ("must", "must_not"):
        pred_dict = getattr(pred, slot)
        gold_dict = getattr(gold, slot)
        for dim in DIMENSIONS:
            pred_val = _to_set(pred_dict.get(dim))
            gold_val = _to_set(gold_dict.get(dim))
            if pred_val != gold_val:
                return False
    return True


def negation_f1(pred: ConditionResult, gold: ConditionResult) -> dict:
    """
    只計算 must_not 的整體 F1（跨所有維度的 micro 平均）。

    參數：
        pred: 預測的 ConditionResult
        gold: 黃金標準的 ConditionResult

    回傳：
        dict：{"p": float, "r": float, "f1": float}
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for dim in DIMENSIONS:
        pred_val = _to_set(pred.must_not.get(dim))
        gold_val = _to_set(gold.must_not.get(dim))

        # 都是空的維度不計入（避免分母膨脹）
        if not pred_val and not gold_val:
            continue

        tp = len(pred_val & gold_val)
        fp = len(pred_val - gold_val)
        fn = len(gold_val - pred_val)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"p": precision, "r": recall, "f1": f1}


def aggregate_metrics(results: list[dict]) -> dict:
    """
    聚合多筆查詢的評估結果，計算各指標的平均值。

    參數：
        results: 每筆查詢的評估結果列表，每個元素為 dict，
                 通常包含 slot_f1 的結果、exact_match、negation_f1 等欄位。
                 預期格式：
                 [
                   {
                     "slot_f1": {"must": {...}, "must_not": {...}, "overall": {...}},
                     "exact_match": bool,
                     "negation_f1": {"p": float, "r": float, "f1": float},
                   },
                   ...
                 ]

    回傳：
        dict：各指標的平均值
        {
            "slot_f1_overall": float,       # slot_f1 整體平均
            "exact_match_rate": float,      # exact match 比率（0.0~1.0）
            "negation_f1": {"p": float, "r": float, "f1": float},
            "must_dim_avg": {"course_lang": float, ...},   # 各維度平均 F1
            "must_not_dim_avg": {"course_lang": float, ...},
            "n": int,                       # 樣本數
        }
    """
    if not results:
        return {
            "slot_f1_overall": 0.0,
            "exact_match_rate": 0.0,
            "negation_f1": {"p": 0.0, "r": 0.0, "f1": 0.0},
            "must_dim_avg": {d: 0.0 for d in DIMENSIONS},
            "must_not_dim_avg": {d: 0.0 for d in DIMENSIONS},
            "n": 0,
        }

    n = len(results)

    # 聚合 slot_f1 overall
    slot_f1_overall_sum = 0.0
    for r in results:
        sf = r.get("slot_f1", {})
        slot_f1_overall_sum += sf.get("overall", {}).get("f1", 0.0)

    # 聚合 exact_match
    em_sum = sum(1 for r in results if r.get("exact_match", False))

    # 聚合 negation_f1
    neg_p_sum = sum(r.get("negation_f1", {}).get("p", 0.0) for r in results)
    neg_r_sum = sum(r.get("negation_f1", {}).get("r", 0.0) for r in results)
    neg_f1_sum = sum(r.get("negation_f1", {}).get("f1", 0.0) for r in results)

    # 聚合各維度平均 F1
    must_dim_sums: dict[str, float] = {d: 0.0 for d in DIMENSIONS}
    must_not_dim_sums: dict[str, float] = {d: 0.0 for d in DIMENSIONS}

    for r in results:
        sf = r.get("slot_f1", {})
        for dim in DIMENSIONS:
            must_dim_sums[dim] += sf.get("must", {}).get(dim, {}).get("f1", 0.0)
            must_not_dim_sums[dim] += sf.get("must_not", {}).get(dim, {}).get("f1", 0.0)

    return {
        "slot_f1_overall": slot_f1_overall_sum / n,
        "exact_match_rate": em_sum / n,
        "negation_f1": {
            "p": neg_p_sum / n,
            "r": neg_r_sum / n,
            "f1": neg_f1_sum / n,
        },
        "must_dim_avg": {d: must_dim_sums[d] / n for d in DIMENSIONS},
        "must_not_dim_avg": {d: must_not_dim_sums[d] / n for d in DIMENSIONS},
        "n": n,
    }
