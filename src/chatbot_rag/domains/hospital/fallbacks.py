"""
屏東基督教醫院領域專屬 Fallback 回應。

當系統無法正常處理時使用的預設回應。
"""

from typing import Any, Dict

# ============================================================================
# Fallback 回應定義
# ============================================================================

FALLBACK_RESPONSES: Dict[str, Dict[str, Any]] = {
    # 一般錯誤回應
    "general_error": {
        "zh-hant": """不好意思，系統暫時無法處理您的請求。

請稍後再試，或撥打客服專線 **08-7368686** 由專人為您服務。

祝您順心！""",
        "zh-hans": """不好意思，系统暂时无法处理您的请求。

请稍后再试，或拨打客服专线 **08-7368686** 由专人为您服务。

祝您顺心！""",
        "en": """I apologize, but the system is temporarily unable to process your request.

Please try again later, or call our customer service hotline **08-7368686** for assistance.

Wishing you well!""",
    },

    # 查無資料回應
    "no_results": {
        "zh-hant": """抱歉，您的問題我目前查不到那麼細的資料，有可能是資訊還未完全上線，也可能您的問題需要更專業的單位說明～

建議您前往 [屏基官網](https://www.ptch.org.tw/index.php/index) 查詢，或致電客服專線：☎️ **08-7368686**

祝您一切順利！""",
        "zh-hans": """抱歉，您的问题我目前查不到那么细的资料，有可能是信息还未完全上线，也可能您的问题需要更专业的单位说明～

建议您前往 [屏基官网](https://www.ptch.org.tw/index.php/index) 查询，或致电客服专线：☎️ **08-7368686**

祝您一切顺利！""",
        "en": """I'm sorry, but I couldn't find detailed information for your question. The information might not be fully online yet, or your question may require a more specialized department to explain.

I suggest visiting the [PTCH official website](https://www.ptch.org.tw/index.php/index) for more information, or calling our customer service hotline: ☎️ **08-7368686**

Wishing you all the best!""",
    },

    # 個資問題回應
    "privacy_inquiry": {
        "zh-hant": """感謝您的提問！😊

關於您詢問的個人醫療資訊（如病歷、看診記錄、費用明細等），為保護您的隱私權益，這些資訊需要透過正式管道查詢：

## 🔒 查詢方式

1. **親自至醫院**：攜帶身分證件至服務台或病歷室申請
2. **電話洽詢**：撥打客服專線 **08-7368686**，將有專人協助

## 📍 服務時間
週一至週五 08:00-17:00

若您有其他關於醫院服務、掛號、門診的問題，我很樂意為您解答喔！

祝您身體健康！✨""",
        "zh-hans": """感谢您的提问！😊

关于您询问的个人医疗信息（如病历、看诊记录、费用明细等），为保护您的隐私权益，这些信息需要通过正式渠道查询：

## 🔒 查询方式

1. **亲自至医院**：携带身份证件至服务台或病历室申请
2. **电话咨询**：拨打客服专线 **08-7368686**，将有专人协助

## 📍 服务时间
周一至周五 08:00-17:00

若您有其他关于医院服务、挂号、门诊的问题，我很乐意为您解答喔！

祝您身体健康！✨""",
        "en": """Thank you for your question! 😊

Regarding the personal medical information you inquired about (such as medical records, visit history, billing details, etc.), to protect your privacy rights, this information must be obtained through official channels:

## 🔒 How to Inquire

1. **Visit the hospital in person**: Bring your ID to the service desk or medical records office
2. **Call us**: Contact our customer service hotline **08-7368686** for assistance

## 📍 Service Hours
Monday to Friday 08:00-17:00

If you have other questions about hospital services, appointments, or outpatient clinics, I'd be happy to help!

Wishing you good health! ✨""",
    },

    # 離題問題回應
    "out_of_scope": {
        "zh-hant": """謝謝您的提問！😊

我是屏東基督教醫院的服務小天使，專門協助您解答與醫院服務相關的問題，例如：

- 📋 **掛號流程**與**門診時間**
- 🩺 **各科別服務**諮詢
- 🏥 **就醫須知**與**院內設施**
- ☎️ **聯絡方式**與**交通資訊**

您詢問的內容可能超出我的服務範圍，但如果您有任何健康或就醫相關的問題，我很樂意為您服務！

祝您身體健康！✨""",
        "zh-hans": """谢谢您的提问！😊

我是屏东基督教医院的服务小天使，专门协助您解答与医院服务相关的问题，例如：

- 📋 **挂号流程**与**门诊时间**
- 🩺 **各科别服务**咨询
- 🏥 **就医须知**与**院内设施**
- ☎️ **联系方式**与**交通信息**

您询问的内容可能超出我的服务范围，但如果您有任何健康或就医相关的问题，我很乐意为您服务！

祝您身体健康！✨""",
        "en": """Thank you for your question! 😊

I'm the service assistant at Pingtung Christian Hospital, here to help you with hospital-related questions, such as:

- 📋 **Registration process** and **clinic hours**
- 🩺 **Department services** consultation
- 🏥 **Patient guide** and **hospital facilities**
- ☎️ **Contact information** and **transportation**

Your question may be outside my service scope, but if you have any health or medical-related questions, I'd be happy to help!

Wishing you good health! ✨""",
    },

    # 系統繁忙回應
    "system_busy": {
        "zh-hant": """系統目前較為繁忙，請稍後再試。

如需緊急協助，請撥打客服專線 **08-7368686**。

感謝您的耐心等候！""",
        "zh-hans": """系统目前较为繁忙，请稍后再试。

如需紧急协助，请拨打客服专线 **08-7368686**。

感谢您的耐心等候！""",
        "en": """The system is currently busy, please try again later.

For urgent assistance, please call our customer service hotline **08-7368686**.

Thank you for your patience!""",
    },
}


def get_fallback_response(
    response_type: str,
    language: str = "zh-hant",
) -> str:
    """
    取得指定類型和語言的 fallback 回應。

    Args:
        response_type: 回應類型（如 "privacy_inquiry", "out_of_scope"）
        language: 語言代碼（如 "zh-hant", "en"）

    Returns:
        對應的 fallback 回應文字
    """
    if response_type not in FALLBACK_RESPONSES:
        response_type = "general_error"

    responses = FALLBACK_RESPONSES[response_type]

    # 語言 fallback: zh-hans -> zh-hant, other -> en -> zh-hant
    if language in responses:
        return responses[language]
    if language == "zh-hans" and "zh-hant" in responses:
        return responses["zh-hant"]
    if "en" in responses:
        return responses["en"]
    return responses.get("zh-hant", "")
