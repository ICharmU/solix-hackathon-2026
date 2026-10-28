"""
Audio transcription module using OpenAI Whisper API with timestamped segment alignment.
Includes segment chunking for arbitrary duration audio files.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from src.schemas import AudioSegment

logger = logging.getLogger(__name__)

# OpenAI audio upload limit is 25 MB; chunk at ~20 MB
MAX_AUDIO_CHUNK_BYTES = 20 * 1024 * 1024

class AudioTranscriber:
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required for Whisper transcription.")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def transcribe(self, audio_path: Optional[Path], time_offset_sec: float = 0.0) -> List[AudioSegment]:
        """
        Transcribes the given WAV audio file into timestamped AudioSegment objects.
        Returns empty list if audio_path is None or file does not exist / is empty.
        """
        if not audio_path or not audio_path.exists():
            logger.info("No audio file provided or file does not exist. Returning empty transcript.")
            return []

        file_size = audio_path.stat().st_size
        if file_size < 1000:  # Less than 1KB is effectively empty/header only
            logger.info("Audio file is empty or near-silent header. Returning empty transcript.")
            return []

        logger.info(f"Transcribing audio ({file_size / (1024*1024):.2f} MB) using {self.model}...")

        try:
            with open(audio_path, "rb") as f:
                # Request verbose_json to obtain segment-level start/end timestamps
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            segments: List[AudioSegment] = []
            raw_segments = getattr(response, "segments", None)

            if raw_segments:
                for seg in raw_segments:
                    start = float(seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0))
                    end = float(seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0))
                    text = str(seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")).strip()

                    if text:
                        segments.append(AudioSegment(
                            start=round(start + time_offset_sec, 3),
                            end=round(end + time_offset_sec, 3),
                            text=text
                        ))
            elif getattr(response, "text", None):
                # Fallback if segments are not explicitly broken down
                full_text = response.text.strip()
                if full_text:
                    segments.append(AudioSegment(
                        start=time_offset_sec,
                        end=time_offset_sec + 1.0,
                        text=full_text
                    ))

            logger.info(f"Transcription complete: extracted {len(segments)} timestamped speech segments.")
            return segments

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

