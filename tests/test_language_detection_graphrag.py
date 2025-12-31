from chatbot_graphrag.graph_workflow.nodes.normalize import detect_language


def test_detect_language_ja_kana_priority():
    assert detect_language("これは何ですか？") == "ja"


def test_detect_language_ko_hangul_priority():
    assert detect_language("예약은 어떻게 하나요?") == "ko"


def test_detect_language_zh_tw_traditional():
    assert detect_language("請問門診時間是幾點？") == "zh-TW"


def test_detect_language_zh_cn_simplified():
    assert detect_language("请问门诊时间是几点？") == "zh-CN"


def test_detect_language_en_default():
    assert detect_language("What are your visiting hours?") == "en"


def test_detect_language_mixed_prefers_en_when_mostly_english():
    # Contains a small CJK hospital name but question is primarily English
    assert detect_language("PTCH 屏基 visiting hours?") == "en"


def test_detect_language_mixed_prefers_zh_when_enough_cjk_signal():
    assert detect_language("PTCH 屏基 掛號流程怎麼走") == "zh-TW"


