"""
Data schemas for multimodal temporal alignment, frame extraction, transcription,
Tier 1 visual descriptors, and text embeddings.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class AudioSegment(BaseModel):
    """Timestamped audio transcription segment."""
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Transcribed speech text")

class FrameDescriptor(BaseModel):
    """Structured Tier 1 multimodal scene descriptor produced by gpt-4o-mini."""
    active_application: str = Field(default="", description="Primary active application in focus")
    window_title: str = Field(default="", description="Title bar, document name, or active browser tab")
    schema_context: str = Field(default="", description="Visible tables, column headers, form fields, and null/empty indicators")
    observed_interaction: str = Field(default="", description="User focus or interaction (e.g. typing, clicking, modal prompt)")
    background_apps: List[str] = Field(default_factory=list, description="Other visible background apps, windows, or tabs")

    def to_raw_text(self, audio_context: str = "") -> str:
        """Converts structured descriptor into a dense string representation for embedding."""
        parts = [
            f"App: {self.active_application or 'Unknown'}",
            f"Title: {self.window_title or 'Untitled'}",
            f"Context: {self.schema_context or 'None'}",
            f"Action: {self.observed_interaction or 'Idle'}"
        ]
        if self.background_apps:
            parts.append(f"Background: {', '.join(self.background_apps)}")
        if audio_context:
            parts.append(f"Audio: {audio_context}")
        return " | ".join(parts)

class TimelineSlot(BaseModel):
    """1-second aligned timeline slot combining visual frame, audio, descriptor, and embedding."""
    timestamp: int = Field(..., description="Second index from start of recording")
    frame_path: str = Field(..., description="Relative or absolute path to frame image")
    has_audio: bool = Field(default=False, description="Whether speech was detected in this interval")
    audio_text: str = Field(default="", description="Transcribed speech within this 1s window")
    is_static: bool = Field(default=False, description="Whether frame is unchanged from prior frame")
    diff_score: float = Field(default=0.0, description="Normalized pixel difference relative to previous frame")

    # Tier 1 outputs
    descriptor: Optional[FrameDescriptor] = Field(
        default=None,
        description="Structured gpt-4o-mini visual state descriptor"
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Flattened dense descriptor text used for vector embedding"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Dense vector embedding generated from raw_text"
    )

class SessionMetadata(BaseModel):
    """Metadata describing the ingested video session."""
    session_id: str
    source_video: str
    duration_seconds: float
    total_frames: int
    fps: int = 1
    has_audio_track: bool
    created_at: str

class TimelineManifest(BaseModel):
    """Master manifest capturing the synchronized multimodal timeline with descriptors and embeddings."""
    metadata: SessionMetadata
    audio_segments: List[AudioSegment] = Field(default_factory=list)
    slots: List[TimelineSlot] = Field(default_factory=list)
