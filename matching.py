"""Score free-text primary diagnoses against the locked 60-case reference labels."""

from __future__ import annotations

import math
import re
from typing import Optional

NEGATION_MARKERS = (
    "未见",
    "否定",
    "不支持",
    "不考虑",
    "不像",
    "排除",
    "no evidence",
    "without",
    "not consistent",
    "unlikely",
)

GOLD_KEYWORDS: dict[str, list[str]] = {
    "气胸/胸腔积气": ["气胸", "胸腔积气", "胸膜积气", "pneumothorax", "pleural air", "pleural gas"],
    "肺炎": [
        "肺炎",
        "支气管肺炎",
        "大叶性肺炎",
        "肺部感染",
        "感染性肺实变",
        "炎性浸润",
        "肺炎性",
        "pneumonia",
        "bronchopneumonia",
        "lobar pneumonia",
        "pulmonary infection",
    ],
    "肺癌": [
        "肺癌",
        "支气管肺癌",
        "中央型",
        "周围型",
        "肺恶性肿瘤",
        "肺恶性占位",
        "lung cancer",
        "bronchogenic",
        "pulmonary malignancy",
        "malignant pulmonary",
    ],
    "支气管炎": ["支气管炎", "气管支气管炎", "支气管感染", "bronchitis", "tracheobronchitis"],
    "泌尿系结石/肾结石": [
        "肾结石",
        "泌尿系结石",
        "尿路结石",
        "结石",
        "nephrolithiasis",
        "renal calculus",
        "renal stone",
        "urolithiasis",
    ],
    "泌尿系结石/肾输尿管结石": [
        "肾输尿管结石",
        "输尿管结石",
        "肾结石",
        "泌尿系结石",
        "尿路结石",
        "结石",
        "ureterolithiasis",
        "ureteral calculus",
        "renal calculus",
        "urolithiasis",
    ],
    "泌尿系结石/输尿管结石": [
        "输尿管结石",
        "泌尿系结石",
        "尿路结石",
        "肾结石",
        "结石",
        "ureterolithiasis",
        "ureteral calculus",
        "urolithiasis",
    ],
    "肠梗阻": ["肠梗阻", "机械性肠梗阻", "小肠梗阻", "结肠梗阻", "obstruction", "ileus"],
    "空腔脏器穿孔": [
        "穿孔",
        "气腹",
        "腹腔游离气体",
        "膈下游离气体",
        "腹腔积气",
        "perforation",
        "pneumoperitoneum",
        "free air",
        "free intraperitoneal",
    ],
    "未见明显异常": [
        "未见明显异常",
        "未见异常",
        "未见急性异常",
        "未见明显急腹症",
        "正常",
        "阴性",
        "无异常",
        "非梗阻性肠气",
        "肠气分布",
        "急腹症相关征象阴性",
        "no significant abnormality",
        "unremarkable",
        "normal",
        "negative",
        "no acute",
    ],
}


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    return re.sub(r"[\s,，、/\\\-\_\(\)（）\[\]:：;；.。·\"'“”‘’hH]", "", text)


def _negated(normalized_primary: str, keyword_norm: str) -> bool:
    idx = normalized_primary.find(keyword_norm)
    if idx < 0:
        return False
    window = normalized_primary[max(0, idx - 12) : idx]
    return any(normalize_text(m) in window for m in NEGATION_MARKERS if normalize_text(m))


def is_correct(primary: Optional[str], gold_label: Optional[str]) -> bool:
    if gold_label is None or (isinstance(gold_label, float) and math.isnan(gold_label)):
        return False
    if not str(primary).strip() or not str(gold_label).strip():
        return False
    gold_label = str(gold_label).strip()
    primary_n = normalize_text(primary)
    if not primary_n:
        return False
    for keyword in GOLD_KEYWORDS.get(gold_label, []):
        key_n = normalize_text(keyword)
        if len(key_n) >= 2 and key_n in primary_n and not _negated(primary_n, key_n):
            return True
    for part in gold_label.split("/"):
        part_n = normalize_text(part)
        if len(part_n) >= 2 and part_n in primary_n and not _negated(primary_n, part_n):
            return True
    return False
