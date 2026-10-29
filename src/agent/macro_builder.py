"""
AI Macro Formulator: Formulates human-readable desktop macros from natural language queries.
Grounded in live screen OCR telemetry and standard desktop application keyboard shortcuts.
"""

import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from PIL import ImageGrab

from src.agent.ocr_engine import WindowsOCREngine
from src.agent.macro import Macro, MACROS_DIR
from src.config import BASE_DIR

load_dotenv()
logger = logging.getLogger("macro_builder")

SYSTEM_PROMPT = """You are an expert Desktop Automation Engineer.
Your task is to convert a natural language query into a clean, human-readable Desktop Macro script.

The macro uses a simple World of Warcraft inspired line-by-line syntax:
/click "Exact Text"         - Left-click a visible UI element by its text label
/doubleclick "Exact Text"   - Double-click target text
/rightclick "Exact Text"    - Right-click target text
/hover "Exact Text"         - Move mouse cursor to target
/type "text to type"        - Type characters into the active element
/press key_or_combo         - Send key or combo (e.g. enter, tab, esc, ctrl+s, ctrl+shift+p, alt+tab)
/wait 0.5s                  - Pause execution for duration (e.g. 0.3s, 1.0s, 500ms)
/assert "Expected Text"     - Verify that specific text appears on screen before proceeding
# Comments start with #

Guidelines:
1. Always prefer matching visible text elements provided in the Visible UI Elements context.
2. Add realistic settle delays (e.g. /wait 0.3s) between opening windows, tabs, or menus.
3. Keep the macro concise and deterministic.
4. Output ONLY the raw macro text starting with # Macro: <name>. Do not include markdown code fences (```) or conversational preamble."""


class MacroFormulator:
    """
    Formulates executable desktop macros from natural language instructions.
    """

    def __init__(self, ocr_engine: Optional[WindowsOCREngine] = None):
        self.ocr_engine = ocr_engine or WindowsOCREngine()

    def formulate_macro(
        self,
        query: str,
        name: Optional[str] = None,
        inspect_live_screen: bool = True,
    ) -> Macro:
        """
        Formulates a Macro from a natural language query.
        """
        visible_elements = []
        if inspect_live_screen:
            try:
                temp_img = BASE_DIR / "data" / "cache" / "macro_inspect_screen.jpg"
                temp_img.parent.mkdir(parents=True, exist_ok=True)
                screen = ImageGrab.grab()
                screen.save(temp_img, format="JPEG", quality=85)
                boxes = self.ocr_engine.recognize_image(temp_img)
                # Take unique text tokens with length >= 2
                unique_texts = []
                seen = set()
                for b in boxes:
                    t = b.text.strip()
                    if len(t) >= 2 and t.lower() not in seen:
                        seen.add(t.lower())
                        unique_texts.append(t)
                visible_elements = unique_texts[:60]
            except Exception as e:
                logger.warning(f"Could not inspect live screen for UI context: {e}")

        macro_name = name or self._slugify(query)

        # Attempt LLM synthesis with OpenAI if key is present
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and "your_" not in api_key.lower() and api_key.strip():
            try:
                macro_text = self._call_openai(query, visible_elements, macro_name)
                return Macro.from_text(macro_text, name=macro_name)
            except Exception as e:
                logger.warning(f"OpenAI macro formulation failed ({e}). Falling back to rule engine.")

        # Deterministic rule-based fallback
        macro_text = self._rule_based_formulation(query, visible_elements, macro_name)
        return Macro.from_text(macro_text, name=macro_name)

    def _call_openai(self, query: str, visible_elements: List[str], macro_name: str) -> str:
        """Calls gpt-4o-mini to formulate the macro."""
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        user_content = f"User Request: {query}\n"
        if visible_elements:
            user_content += f"Visible UI Elements currently on user's screen:\n{', '.join(visible_elements[:50])}\n"
        user_content += f"Suggested Macro Name: {macro_name}\nFormulate the best macro."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=400,
        )

        content = response.choices[0].message.content.strip()
        # Clean any accidental markdown fences
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```$", "", content)
        return content

    def _rule_based_formulation(self, query: str, visible_elements: List[str], macro_name: str) -> str:
        """Deterministic fallback formulation based on query patterns."""
        lines = [f"# Macro: {macro_name}", f"# Formulated from query: '{query}'"]
        q_lower = query.lower()

        # Check for tab or button click target (whole-word match, longest first)
        target = None
        for elem in sorted(visible_elements, key=len, reverse=True):
            if len(elem) >= 3 and re.search(rf"\b{re.escape(elem.lower())}\b", q_lower):
                target = elem
                break

        if not target:
            # Extract target after action verbs (ordered from longest to shortest)
            match = re.search(r'(?:switch to|click on|click|open|to|on)\s+["\']?([a-zA-Z0-9_\-\.]+)["\']?', query, re.IGNORECASE)
            if match:
                target = match.group(1)

        if target:
            lines.append(f'/click "{target}"')
            lines.append("/wait 0.3s")

        if "format" in q_lower:
            lines.append("/press ctrl+shift+i")
            lines.append("/wait 0.2s")

        if "save" in q_lower:
            lines.append("/press ctrl+s")

        if "run" in q_lower or "terminal" in q_lower:
            lines.append('/click "Terminal"')
            lines.append("/wait 0.3s")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _slugify(text: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9\s_]", "", text)
        slug = "_".join(clean.strip().lower().split()[:4])
        return slug or "custom_macro"
