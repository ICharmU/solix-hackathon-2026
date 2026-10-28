"""
Tier 1 Multimodal Descriptor Extraction & Embedding Generation.
Extracts structured UI/state context from 1 FPS frames using gpt-4o-mini
and computes dense vector embeddings using text-embedding-3-small.
"""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI

from src.config import BASE_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, VLM_MODEL
from src.schemas import FrameDescriptor, TimelineManifest, TimelineSlot

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert system activity and UI analyzer for computer workflow tracking.
Analyze this screen capture frame from a user's computer session.
Do NOT transcribe raw row-level cell values or full document bodies.
Focus strictly on identifying:
1. active_application: The primary foreground application (e.g., 'Excel', 'Google Chrome', 'VS Code', 'Slack').
2. window_title: The active window title or document name.
3. schema_context: Observable data structure, column headers, form fields, and whether key fields appear null/empty.
4. observed_interaction: The apparent user action or focus (e.g., cursor hovering on row, typing, modal dialog, menu open).
5. background_apps: Other applications or tabs visible in the screen background or taskbar.

You must output ONLY valid JSON matching this schema:
{
  "active_application": "<string>",
  "window_title": "<string>",
  "schema_context": "<string>",
  "observed_interaction": "<string>",
  "background_apps": ["<string>"]
}
"""

def encode_image_base64(image_path: Path) -> str:
    """Reads image file and converts to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class DescriptorExtractor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = VLM_MODEL,
        detail: str = "low"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.detail = detail
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required for descriptor extraction.")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def extract_descriptor(self, image_path: Path, timestamp: int, audio_text: str = "") -> FrameDescriptor:
        """Calls gpt-4o-mini with frame image to extract structured UI state."""
        b64_image = encode_image_base64(image_path)
        user_message_text = f"Analyze screen frame at second {timestamp}."
        if audio_text:
            user_message_text += f" Associated desktop audio transcript: '{audio_text}'."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                            "detail": self.detail
                        }
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=250
            )
            raw_content = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)

            bg_apps = parsed.get("background_apps", [])
            if isinstance(bg_apps, str):
                bg_apps = [bg_apps]

            return FrameDescriptor(
                active_application=str(parsed.get("active_application", "")),
                window_title=str(parsed.get("window_title", "")),
                schema_context=str(parsed.get("schema_context", "")),
                observed_interaction=str(parsed.get("observed_interaction", "")),
                background_apps=bg_apps
            )

        except Exception as e:
            logger.error(f"Error extracting descriptor for frame at t={timestamp}: {e}")
            return FrameDescriptor(
                active_application="Error",
                window_title="Extraction Failed",
                schema_context=str(e),
                observed_interaction="Error",
                background_apps=[]
            )

class EmbeddingGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = EMBEDDING_MODEL):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required for embedding generation.")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generates dense vector embeddings for a list of text strings in batches."""
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Replace empty strings with a minimal placeholder
            cleaned_batch = [t if t.strip() else "Empty frame" for t in batch]
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_batch
            )
            # Ensure sorted order matching input
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            embeddings.extend(batch_embeddings)

        return embeddings

def enrich_manifest(
    manifest: TimelineManifest,
    base_dir: Path = BASE_DIR,
    use_cache_for_static: bool = True,
    max_workers: int = 5,
    detail: str = "low",
    mock: bool = False
) -> TimelineManifest:
    """
    Enriches a TimelineManifest by generating gpt-4o-mini descriptors,
    raw text representations, and vector embeddings for every timeline slot.
    """
    slots = manifest.slots
    total_slots = len(slots)
    logger.info(f"Enriching {total_slots} slots with gpt-4o-mini descriptors and embeddings...")

    if mock:
        logger.info("Mock mode enabled: generating deterministic mock descriptors and embeddings.")
        for slot in slots:
            slot.descriptor = FrameDescriptor(
                active_application="Mock App",
                window_title="Mock Window",
                schema_context="Mock Schema Context",
                observed_interaction="Mock Interaction",
                background_apps=["Mock BG"]
            )
            slot.raw_text = slot.descriptor.to_raw_text(slot.audio_text)
            slot.embedding = [0.05 * (slot.timestamp % 5)] * 64
        return manifest

    extractor = DescriptorExtractor(detail=detail)

    # 1. Identify which slots require VLM inference
    # If a slot is static and use_cache_for_static is enabled, we can reuse the previous descriptor
    indices_to_query: List[int] = []
    for idx, slot in enumerate(slots):
        if idx == 0 or not use_cache_for_static or not slot.is_static:
            indices_to_query.append(idx)

    logger.info(
        f"Selected {len(indices_to_query)}/{total_slots} frames for VLM inference "
        f"({total_slots - len(indices_to_query)} static frames cached)."
    )

    # 2. Concurrently extract descriptors for unique/changing frames
    extracted_descriptors: Dict[int, FrameDescriptor] = {}

    def _process_slot(idx: int):
        s = slots[idx]
        frame_abs_path = (base_dir / s.frame_path).resolve()
        desc = extractor.extract_descriptor(
            image_path=frame_abs_path,
            timestamp=s.timestamp,
            audio_text=s.audio_text
        )
        return idx, desc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_slot, idx): idx for idx in indices_to_query}
        for future in as_completed(futures):
            idx, desc = future.result()
            extracted_descriptors[idx] = desc

    # 3. Propagate descriptors to all slots (including static ones) and compute raw_text
    raw_texts: List[str] = []
    last_descriptor: Optional[FrameDescriptor] = None

    for idx, slot in enumerate(slots):
        if idx in extracted_descriptors:
            last_descriptor = extracted_descriptors[idx]
        elif slot.is_static and last_descriptor:
            # Reusing last known descriptor
            pass
        else:
            # Fallback if something failed
            last_descriptor = extracted_descriptors.get(idx, FrameDescriptor(
                active_application="Unknown",
                window_title="Unknown",
                schema_context="None",
                observed_interaction="Idle",
                background_apps=[]
            ))

        slot.descriptor = last_descriptor
        slot.raw_text = last_descriptor.to_raw_text(slot.audio_text)
        raw_texts.append(slot.raw_text)

    # 4. Generate batch embeddings for all raw_texts
    logger.info(f"Generating dense vector embeddings for all {len(raw_texts)} slots...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate_embeddings(raw_texts)

    for slot, emb in zip(slots, embeddings):
        slot.embedding = emb

    logger.info("Successfully enriched all slots with descriptors, raw_text, and embeddings.")
    return manifest

