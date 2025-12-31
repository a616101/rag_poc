"""
語言正規化節點

偵測語言並正規化問題以供處理。

主要功能：
- 語言偵測（zh-TW, zh-CN, en, ja, ko）
- 問題文字正規化
- 關鍵詞抽取
"""

import logging
import re
import unicodedata
from typing import Any

from chatbot_graphrag.graph_workflow.types import GraphRAGState

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """
    偵測文字的語言。

    Returns:
        語言代碼（例如 "zh-TW", "en", "ja"）
    """
    # 注意：
    # - 我們故意忽略標點符號/空白以避免比例失真。
    # - 我們不依賴外部語言偵測庫（無額外依賴）。
    # - 優先順序：日語（假名）> 韓語（韓文）> 中文（CJK）> 英語（拉丁/其他）。

    if not text or not text.strip():
        return "en"

    # 用於區分繁體中文和簡體中文的小型但實用的字元集。
    # 如果兩個集合都沒出現，我們預設為 zh-TW，因為此專案以 zh-TW 為主。
    simplified_only = set(
        "们么为这来时里没还问国发进东门龙云电气学后师医药页语员愿远运证织"
        "无线乡写严听会开关间价应实术书万与专业车连两边达过欢换记际见举据"
        "亲却让热认软声条孙钟"
    )
    traditional_only = set(
        "們麼為這來時裡沒還問國發進東門龍雲電氣學後師醫藥頁語員願遠運證織"
        "無線鄉寫嚴聽會開關間價應實術書萬與專業車連兩邊達過歡換記際見舉據"
        "親卻讓熱認軟聲條孫鐘"
    )

    has_hiragana = False
    has_katakana = False
    has_hangul = False
    cjk_count = 0
    latin_count = 0
    simp_count = 0
    trad_count = 0

    for ch in text:
        # 跳過空白、標點符號、符號和數字
        cat = unicodedata.category(ch)
        if cat[0] in {"Z", "P", "S", "N"}:
            continue

        # 平假名：U+3040 - U+309F
        if "\u3040" <= ch <= "\u309f":
            has_hiragana = True
            continue

        # 片假名：U+30A0 - U+30FF
        if "\u30a0" <= ch <= "\u30ff":
            has_katakana = True
            continue

        # 韓文：U+AC00 - U+D7AF，加上韓文字母 U+1100 - U+11FF
        if "\uac00" <= ch <= "\ud7af" or "\u1100" <= ch <= "\u11ff":
            has_hangul = True
            continue

        # CJK 統一漢字：U+4E00 - U+9FFF
        if "\u4e00" <= ch <= "\u9fff":
            cjk_count += 1
            if ch in simplified_only:
                simp_count += 1
            if ch in traditional_only:
                trad_count += 1
            continue

        # 拉丁字母（基本 + 擴展），用於混合文字中偏好 "en"
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            latin_count += 1
            continue
        if cat[0] == "L":
            # 盡力處理擴展拉丁字母（例如越南語/歐洲語言名稱）
            try:
                if "LATIN" in unicodedata.name(ch):
                    latin_count += 1
            except ValueError:
                pass

    # 優先順序：假名 > 韓文
    if has_hiragana or has_katakana:
        return "ja"
    if has_hangul:
        return "ko"

    # 混合文字中的中文 vs 英文：
    # 如果 CJK 存在有意義，則視為中文；否則預設為英文。
    if cjk_count > 0:
        if latin_count == 0:
            is_chinese = True
        else:
            # 要求足夠的 CJK 信號，以避免將大部分英文的問題錯誤分類
            ratio = cjk_count / (cjk_count + latin_count)
            is_chinese = cjk_count >= 3 and ratio >= 0.2

        if is_chinese:
            if simp_count > trad_count:
                return "zh-CN"
            # 對於平局/未知情況預設為 zh-TW，因為專案主要是 zh-TW
            return "zh-TW"

    return "en"


def normalize_question(text: str) -> str:
    """
    正規化問題文字。

    - 移除過多空白
    - 正規化標點符號
    - 修剪到合理長度
    """
    # 移除過多空白
    text = re.sub(r"\s+", " ", text.strip())

    # 正規化常見的標點符號變體
    text = text.replace("？", "?")
    text = text.replace("！", "!")
    text = text.replace("。", ".")
    text = text.replace("，", ",")

    # 確保問題以標點符號結尾
    if text and text[-1] not in ".?!":
        text = text + "?"

    # 限制長度
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text


def extract_key_terms(text: str) -> list[str]:
    """
    從問題中抽取關鍵詞以改善檢索。

    Returns:
        關鍵詞列表
    """
    # 移除常見的問題詞
    stop_words_en = {
        "what", "how", "when", "where", "why", "who", "which",
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "should", "will",
        "the", "a", "an", "in", "on", "at", "to", "for",
    }
    stop_words_zh = {
        "什麼", "怎麼", "如何", "哪裡", "為什麼", "誰", "哪個",
        "是", "有", "可以", "能", "會", "要", "的", "嗎", "呢",
    }

    # 分割文字
    words = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())

    # 過濾停用詞和短詞
    key_terms = [
        w for w in words
        if w not in stop_words_en
        and w not in stop_words_zh
        and len(w) > 1
    ]

    return key_terms[:10]  # 限制為前 10 個詞


async def normalize_node(state: GraphRAGState) -> dict[str, Any]:
    """
    用於語言偵測和問題正規化的節點。

    Returns:
        更新後的狀態，包含 normalized_question 和 user_language
    """
    import time
    from chatbot_graphrag.graph_workflow.nodes.status import emit_status

    emit_status("normalize", "START")

    start_time = time.time()
    question = state.get("question", "")
    retrieval_path = list(state.get("retrieval_path", []))
    timing = dict(state.get("timing", {}))

    # 偵測語言
    user_language = detect_language(question)
    logger.debug(f"偵測到語言: {user_language}")

    # 正規化問題
    normalized = normalize_question(question)
    logger.debug(f"正規化問題: {normalized[:100]}...")

    # 抽取關鍵詞
    key_terms = extract_key_terms(question)
    logger.debug(f"關鍵詞: {key_terms}")

    emit_status("normalize", "DONE")
    return {
        "normalized_question": normalized,
        "user_language": user_language,
        "retrieval_path": retrieval_path + [f"normalize:{user_language}"],
        "timing": {**timing, "normalize_ms": (time.time() - start_time) * 1000},
    }
