"""
Responses API 串流累加器

用於收集和累加 OpenAI Responses API 的串流事件，
計算最終完整文字、使用量和時間統計。

主要類別：
- ChannelBuffer: 單一頻道的緩衝區
- ResponsesAccumulator: 多頻道串流事件累加器
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCompletedEvent,
    ResponseUsage,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
)


@dataclass
class ChannelBuffer:
    """
    單一頻道的緩衝區。

    用於累積串流 delta 事件，追蹤時間戳記，
    並在收到完整內容時覆蓋。
    """
    deltas: List[str] = field(default_factory=list)  # 累積的 delta 列表
    full: Optional[str] = None  # 完整內容（當收到 done 事件時設置）
    first_at: Optional[float] = None  # 第一個 delta 的時間戳記
    last_at: Optional[float] = None  # 最後一個 delta 的時間戳記

    def add_delta(self, text: str) -> None:
        """添加一個 delta 到緩衝區。"""
        now = time.monotonic()
        if self.first_at is None:
            self.first_at = now
        self.last_at = now
        if text:
            self.deltas.append(text)

    def set_full(self, text: str) -> None:
        """設置完整內容（覆蓋累積的 delta）。"""
        self.full = text

    @property
    def text(self) -> str:
        """取得完整文字（優先使用 full，否則合併 deltas）。"""
        if self.full is not None:
            return self.full
        return "".join(self.deltas)

    @property
    def duration_ms(self) -> Optional[float]:
        """計算從第一個到最後一個 delta 的持續時間（毫秒）。"""
        if self.first_at is None or self.last_at is None:
            return None
        return (self.last_at - self.first_at) * 1000.0


@dataclass
class ResponsesAccumulator:
    """收集所有 streaming events，最後算出完整 text + usage + 時間。"""

    channels: Dict[str, ChannelBuffer] = field(
        default_factory=lambda: {
            "output_text": ChannelBuffer(),      # 主要輸出文字
            "reasoning": ChannelBuffer(),         # 推理過程文字
            "reasoning_summary": ChannelBuffer(), # 推理摘要
            "tool_arguments": ChannelBuffer(),    # 工具呼叫參數
            "audio_transcript": ChannelBuffer(),  # 音訊轉錄
        }
    )
    usage: Any = None  # API 使用量資訊
    response_id: Optional[str] = None  # 回應 ID

    started_at: float = field(default_factory=time.monotonic)  # 開始時間
    completed_at: Optional[float] = None  # 完成時間
    meta_emitted: bool = False  # 是否已發送 meta 事件

    def apply_event(self, event: Any) -> None:
        """
        應用一個串流事件到累加器。

        根據事件類型將其分派到相應的頻道。
        """
        # 1) delta 事件 - 累積部分內容
        if isinstance(event, ResponseTextDeltaEvent):
            self.channels["output_text"].add_delta(event.delta)

        elif isinstance(event, ResponseReasoningTextDeltaEvent):
            self.channels["reasoning"].add_delta(event.delta)

        elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            self.channels["reasoning_summary"].add_delta(event.delta)

        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            self.channels["tool_arguments"].add_delta(event.delta)

        elif isinstance(event, ResponseAudioTranscriptDeltaEvent):
            self.channels["audio_transcript"].add_delta(event.delta)

        # 2) done 事件：用完整內容覆蓋
        elif isinstance(event, ResponseTextDoneEvent):
            self.channels["output_text"].add_delta("")
            self.channels["output_text"].set_full(event.text)

        elif isinstance(event, ResponseReasoningTextDoneEvent):
            self.channels["reasoning"].add_delta("")
            self.channels["reasoning"].set_full(event.text)

        elif isinstance(event, ResponseReasoningSummaryTextDoneEvent):
            self.channels["reasoning_summary"].add_delta("")
            self.channels["reasoning_summary"].set_full(event.text)

        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            self.channels["tool_arguments"].add_delta("")
            self.channels["tool_arguments"].set_full(event.arguments)

        elif isinstance(event, ResponseAudioTranscriptDoneEvent):
            self.channels["audio_transcript"].add_delta("")
            self.channels["audio_transcript"].set_full(event.text)

        # 3) usage / id：在 Completed 事件中提取
        if isinstance(event, ResponseCompletedEvent):
            resp = event.response
            self.response_id = getattr(resp, "id", None)
            self.usage = getattr(resp, "usage", None)
            self.completed_at = time.monotonic()

        elif isinstance(event, (ResponseFailedEvent, ResponseIncompleteEvent)):
            resp = getattr(event, "response", None)
            if resp is not None:
                self.response_id = getattr(resp, "id", None)
                self.usage = getattr(resp, "usage", None)
                self.completed_at = time.monotonic()

    def has_content(self) -> bool:
        """檢查是否有任何頻道包含內容。"""
        return any(buf.text for buf in self.channels.values())

    def build_meta(self) -> Dict[str, Any]:
        """
        建構 meta 資訊字典。

        包含所有頻道的完整文字、持續時間和字元數，
        以及 usage 資訊和 response_id。
        """
        self.meta_emitted = True

        channels_meta: Dict[str, Any] = {}
        for name, buf in self.channels.items():
            text = buf.text
            if not text:
                continue
            channels_meta[name] = {
                "text": text,
                "duration_ms": buf.duration_ms,
                "char_count": len(text),
            }

        usage_dict = None
        if self.usage is not None:
            if hasattr(self.usage, "model_dump"):
                usage_dict = self.usage.model_dump(mode="python")
            elif isinstance(self.usage, dict):
                usage_dict = self.usage

        return {
            "response_id": self.response_id,
            "usage": usage_dict,
            "channels": channels_meta,
        }
