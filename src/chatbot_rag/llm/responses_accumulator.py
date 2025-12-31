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
    deltas: List[str] = field(default_factory=list)
    full: Optional[str] = None
    first_at: Optional[float] = None
    last_at: Optional[float] = None

    def add_delta(self, text: str) -> None:
        now = time.monotonic()
        if self.first_at is None:
            self.first_at = now
        self.last_at = now
        if text:
            self.deltas.append(text)

    def set_full(self, text: str) -> None:
        self.full = text

    @property
    def text(self) -> str:
        if self.full is not None:
            return self.full
        return "".join(self.deltas)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.first_at is None or self.last_at is None:
            return None
        return (self.last_at - self.first_at) * 1000.0


@dataclass
class ResponsesAccumulator:
    """收集所有 streaming events，最後算出完整 text + usage + 時間。"""

    channels: Dict[str, ChannelBuffer] = field(
        default_factory=lambda: {
            "output_text": ChannelBuffer(),
            "reasoning": ChannelBuffer(),
            "reasoning_summary": ChannelBuffer(),
            "tool_arguments": ChannelBuffer(),
            "audio_transcript": ChannelBuffer(),
        }
    )
    usage: Any = None
    response_id: Optional[str] = None

    started_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    meta_emitted: bool = False

    def apply_event(self, event: Any) -> None:
        # 1) delta events
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

        # 2) done events：用完整內容覆蓋
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

        # 3) usage / id：在 Completed 事件裡
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
        return any(buf.text for buf in self.channels.values())

    def build_meta(self) -> Dict[str, Any]:
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


