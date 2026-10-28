"""
Configuration settings for the agentic workflow capture and analysis pipeline.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"
AUDIO_DIR = DATA_DIR / "audio"
PROCESSED_DIR = DATA_DIR / "processed"
# Artifact directory: defaults to local data/artifacts if env var not set or invalid
_default_artifact = str(BASE_DIR / "data" / "artifacts")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", _default_artifact))
if not ARTIFACT_DIR.is_absolute():
    ARTIFACT_DIR = BASE_DIR / ARTIFACT_DIR

# Ensure directories exist
for directory in (DATA_DIR, FRAMES_DIR, AUDIO_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Extraction settings
DEFAULT_FPS = 1
DEFAULT_AUDIO_SAMPLE_RATE = 16000  # 16kHz mono recommended for Whisper
DEFAULT_CHUNK_DURATION_SEC = 300   # 5-minute chunk windows for arbitrary duration scaling
STATIC_DIFF_THRESHOLD = 0.008      # Perceptual difference threshold for static frame detection

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

