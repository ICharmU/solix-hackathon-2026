"""
AI Macro Formulator: Formulates human-readable desktop macros from natural language queries.
Grounded in workspace file context, live screen OCR telemetry, and standard desktop keyboard shortcuts.
Employs multi-stage LLM reasoning: Intent Planning -> Context Synthesis -> Pre-Flight Audit.
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
from src.agent.macro import Macro, MacroStep, MACROS_DIR
from src.config import BASE_DIR

load_dotenv()
logger = logging.getLogger("macro_builder")

SYSTEM_PROMPT = """You are an expert Desktop Automation Engineer and Coding Agent.
Your task is to convert a natural language query into a clean, human-readable Desktop Macro script.

The macro uses a World of Warcraft inspired line-by-line syntax:
/click "Exact Text"         - Left-click a visible UI element by its text label
/doubleclick "Exact Text"   - Double-click target text
/rightclick "Exact Text"    - Right-click target text
/hover "Exact Text"         - Move mouse cursor to target
/type "text to type"        - Type characters into the active element
/press key_or_combo         - Send key or combo (e.g. enter, tab, esc, end, home, left, ctrl+s, alt+tab)
/wait 0.5s                  - Pause execution for duration (e.g. 0.3s, 1.0s, 500ms)
/assert "Expected Text"     - Verify that specific text appears on screen before proceeding

Composable Tooling Primitives:
/macro "sub_macro_name"     - Composes an existing macro file from macros/ (e.g. /macro "open_pixi")
/openfile "filename"        - Quick-opens file via Ctrl+P in VS Code / IDE (e.g. /openfile "pixi.toml")
/gotoline line_number       - Jumps directly to specific line in active file via Ctrl+G (preferred for exact lines)
/findtext "search_string"   - Finds text in the active file via Ctrl+F and places cursor on match
/replace "old" "new"        - Replaces text in the active file via Ctrl+H (Ctrl+Alt+Enter to replace all)

CRITICAL RULES FOR CODING & CONFIGURATION TASKS:
1. ENVIRONMENTAL CONTEXT & CLICKING BEFORE MODIFYING:
   - The agent MUST click into the file before applying editing tools.
   - Step 1: Open the target file (compose /macro "open_pixi" if available, or /openfile "<filename>").
   - Step 2: Settle delay (/wait 0.3s) for the file to render.
   - Step 3: CLICK INTO THE FILE on a visible anchor text (/click "<target_text>") to confirm the environmental context and place keyboard focus inside the editor text buffer.
     Example: /click "platforms"
   - Step 4: Move cursor to exact insertion point: /press end, /press left (or /gotoline <line_num>).
   - Step 5: Perform the edit: /type '<addition>' or /replace "old" "new".
   - Step 6: Save the file: /press ctrl+s.
2. Environmental context must always be confirmed: by clicking an anchor text in the file, the agent ensures that the correct file is actively focused and that inputs will land in the file.
3. Use composable sub-macros (/macro) whenever possible for modular workflows.
4. Output ONLY the raw macro text starting with # Macro: <name>. Do not include markdown code fences (```)."""


class MacroFormulator:
    """
    Formulates executable desktop macros from natural language instructions.
    Leverages multi-stage LLM reasoning grounded in workspace file contents.
    """

    def __init__(self, ocr_engine: Optional[WindowsOCREngine] = None):
        self.ocr_engine = ocr_engine or WindowsOCREngine()

    def gather_workspace_context(self, query: str) -> Dict[str, Any]:
        """
        Gathers rich context from the local workspace:
        1. Target and related configuration/code file contents with line numbers.
        2. Available composable sub-macros in macros/.
        3. Project file inventory.
        """
        context: Dict[str, Any] = {
            "target_files": {},
            "available_macros": {},
            "project_files": [],
        }

        # 1. Discover available reusable macros
        if MACROS_DIR.exists():
            for m_path in MACROS_DIR.glob("*.macro"):
                try:
                    context["available_macros"][m_path.stem] = m_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        # 2. Candidate configuration and code files
        common_configs = [
            "pixi.toml",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            ".env",
            "setup.py",
            "requirements.txt",
        ]

        # Files explicitly mentioned in query
        explicit_files = re.findall(
            r'([a-zA-Z0-9_\-\.\/]+\.(?:toml|py|json|md|yaml|yml|js|ts|html|css|txt|sh|ps1))',
            query,
            re.IGNORECASE,
        )

        candidates = list(dict.fromkeys(explicit_files + common_configs))
        q_lower = query.lower()

        # Prioritize files based on query keywords
        if "pixi" in q_lower or "platform" in q_lower or "conda" in q_lower:
            if "pixi.toml" in candidates:
                candidates.remove("pixi.toml")
                candidates.insert(0, "pixi.toml")

        for fname in candidates:
            p = BASE_DIR / fname
            if p.is_file():
                context["project_files"].append(fname)
                try:
                    lines = p.read_text(encoding="utf-8").splitlines()
                    # Include line numbers for exact navigation
                    numbered_lines = [f"{i}: {line}" for i, line in enumerate(lines, start=1)]
                    context["target_files"][fname] = {
                        "total_lines": len(lines),
                        "content_numbered": "\n".join(numbered_lines[:80]),
                        "raw_lines": lines,
                    }
                except Exception as e:
                    logger.debug(f"Could not read {fname}: {e}")

        return context

    def formulate_macro(
        self,
        query: str,
        name: Optional[str] = None,
        inspect_live_screen: bool = True,
    ) -> Macro:
        """
        Formulates a Macro from a natural language query using multi-stage LLM reasoning
        grounded in workspace files and screen anchors.
        """
        # 1. Gather workspace and file context
        context = self.gather_workspace_context(query)

        # 2. Inspect live screen if requested
        visible_elements = []
        if inspect_live_screen:
            try:
                temp_img = BASE_DIR / "data" / "cache" / "macro_inspect_screen.jpg"
                temp_img.parent.mkdir(parents=True, exist_ok=True)
                screen = ImageGrab.grab()
                screen.save(temp_img, format="JPEG", quality=85)
                boxes = self.ocr_engine.recognize_image(temp_img)
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

        # 3. Attempt multi-stage LLM synthesis with OpenAI if key is present
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and "your_" not in api_key.lower() and api_key.strip():
            try:
                macro_text = self._call_openai_multi_stage(query, context, visible_elements, macro_name)
                return Macro.from_text(macro_text, name=macro_name)
            except Exception as e:
                logger.warning(f"OpenAI multi-stage macro formulation failed ({e}). Falling back to context-grounded rule engine.")

        # 4. Context-grounded deterministic rule-based fallback
        macro_text = self._rule_based_formulation(query, visible_elements, macro_name, context)
        return Macro.from_text(macro_text, name=macro_name)

    def _call_openai_multi_stage(
        self,
        query: str,
        context: Dict[str, Any],
        visible_elements: List[str],
        macro_name: str,
    ) -> str:
        """
        Executes a 3-stage LLM reasoning workflow:
        Stage 1: Intent & Target File/Line Grounding
        Stage 2: Macro Synthesis
        Stage 3: Pre-Flight Edit Verification & Keystroke Audit
        """
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # --- Stage 1: Grounded Intent & Target File Analysis ---
        plan = self._call_openai_plan(client, query, context, visible_elements)
        logger.info(f"LLM Stage 1 Plan: {plan}")

        # --- Stage 2: Context-Grounded Macro Synthesis ---
        draft_macro = self._call_openai_synthesize(client, query, plan, context, macro_name)
        logger.info(f"LLM Stage 2 Draft:\n{draft_macro}")

        # --- Stage 3: Pre-Flight Edit Audit & Keystroke Verification ---
        verified_macro = self._call_openai_audit(client, query, draft_macro, context)
        logger.info(f"LLM Stage 3 Verified:\n{verified_macro}")

        return verified_macro

    def _call_openai_plan(self, client: Any, query: str, context: Dict[str, Any], visible_elements: List[str]) -> str:
        """Call 1: Grounded Intent & Target File/Line Analysis."""
        files_summary = ""
        for fname, info in context.get("target_files", {}).items():
            files_summary += f"\n--- File: {fname} (First {min(info['total_lines'], 50)} lines) ---\n"
            files_summary += "\n".join(info["content_numbered"].splitlines()[:50]) + "\n"

        macros_summary = ", ".join(context.get("available_macros", {}).keys()) or "None"

        prompt = f"""User Request: {query}

Available Workspace Context:
Target Files & Line Numbers:
{files_summary}

Existing Reusable Macros in macros/: {macros_summary}
Screen UI Anchors: {', '.join(visible_elements[:30])}

Analyze the request and provide a 2-3 sentence implementation plan:
1. Which file needs editing and on which EXACT line number?
2. What sub-macro (if any) should be composed to open the file?
3. What is the exact insertion or replacement text?"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior desktop automation planner. Identify exact files, line numbers, and composable sub-macros."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def _call_openai_synthesize(
        self,
        client: Any,
        query: str,
        plan: str,
        context: Dict[str, Any],
        macro_name: str,
    ) -> str:
        """Call 2: Context-Grounded Macro Formulation."""
        files_summary = ""
        for fname, info in context.get("target_files", {}).items():
            files_summary += f"\n--- File: {fname} (Line-by-line) ---\n"
            files_summary += "\n".join(info["content_numbered"].splitlines()[:50]) + "\n"

        available_submacros = ""
        for mname, mtext in context.get("available_macros", {}).items():
            available_submacros += f"\n--- Sub-Macro: {mname}.macro ---\n{mtext}\n"

        prompt = f"""User Request: {query}
Implementation Plan: {plan}

Target Files with EXACT line numbers:
{files_summary}

Available Sub-Macros in macros/:
{available_submacros}

Suggested Macro Name: {macro_name}

Generate the Macro script using the WoW-style DSL.
Rules:
- If a sub-macro exists (e.g. /macro "open_pixi"), compose it.
- CRITICAL: Always click into the file on an anchor text before modifying: use /click "<anchor_text>" (e.g. /click "platforms") to verify environmental context and place focus in the editor buffer.
- Use /press end, /press left, and /type '<content>' to edit cleanly without breaking syntax.
- Alternatively use /gotoline <line_num> or /replace "old" "new" if replacing an exact string.
- Always save the file at the end with /press ctrl+s.
- Output ONLY the raw macro lines starting with # Macro: {macro_name}."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```$", "", content)
        return content

    def _call_openai_audit(self, client: Any, query: str, draft_macro: str, context: Dict[str, Any]) -> str:
        """Call 3: Pre-Flight Verification & Keystroke Audit."""
        files_summary = ""
        for fname, info in context.get("target_files", {}).items():
            files_summary += f"\n--- File: {fname} ---\n"
            files_summary += "\n".join(info["content_numbered"].splitlines()[:40]) + "\n"

        prompt = f"""User Request: {query}

Original File Context:
{files_summary}

Draft Macro:
{draft_macro}

Audit the draft macro:
1. Does it click into the file on an anchor text (/click "<text>") to confirm environmental context before modifying?
2. Will the cursor position and typed text produce valid syntax (e.g. valid TOML/JSON/Python)?
3. Are necessary settle delays (/wait 0.3s) present after opening files or modals?
4. Is /press ctrl+s present to save?

If everything is correct, output the macro as-is.
If any adjustment is needed, output the improved macro.
Output ONLY the macro text starting with # Macro:."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```$", "", content)
        return content

    def refine_macro(
        self,
        existing_macro: Macro,
        feedback: str,
    ) -> Macro:
        """Refines an existing macro based on natural language user feedback and file context."""
        context = self.gather_workspace_context(feedback + " " + existing_macro.name)

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and "your_" not in api_key.lower() and api_key.strip():
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                files_summary = ""
                for fname, info in context.get("target_files", {}).items():
                    files_summary += f"\n--- File: {fname} ---\n"
                    files_summary += "\n".join(info["content_numbered"].splitlines()[:50]) + "\n"

                prompt = f"""Existing Macro ({existing_macro.name}):
{existing_macro.to_text()}

User Feedback / Desired Adjustments:
{feedback}

File Context:
{files_summary}

Refine the macro to address the user feedback.
Output ONLY the updated macro text starting with # Macro: {existing_macro.name}."""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                content = response.choices[0].message.content.strip()
                content = re.sub(r"^```[a-zA-Z]*\n", "", content)
                content = re.sub(r"\n```$", "", content)
                return Macro.from_text(content, name=existing_macro.name)
            except Exception as e:
                logger.warning(f"OpenAI refinement failed ({e}). Falling back to rule-based adjustment.")

        return self._rule_based_refinement(existing_macro, feedback, context)

    def _rule_based_refinement(self, existing_macro: Macro, feedback: str, context: Dict[str, Any]) -> Macro:
        """Applies heuristic adjustments when offline."""
        f_lower = feedback.lower()
        new_steps = []

        # Find target line from file context if present
        pixi_info = context.get("target_files", {}).get("pixi.toml")
        target_line = 4
        if pixi_info:
            for idx, line in enumerate(pixi_info["raw_lines"], start=1):
                if line.strip().startswith("platforms ="):
                    target_line = idx
                    break

        has_interaction = any(s.command in ("click", "interact") for s in existing_macro.steps)

        for s in existing_macro.steps:
            if s.command == "findtext" and ("line" in f_lower or "gotoline" in f_lower or "where" in f_lower):
                new_steps.append(MacroStep(line_num=s.line_num, command="gotoline", argument=str(target_line), raw_line=f"/gotoline {target_line}"))
            else:
                new_steps.append(s)

            if s.command in ("openfile", "macro") and ("click" in f_lower or "interact" in f_lower or "focus" in f_lower or "context" in f_lower) and not has_interaction:
                new_steps.append(MacroStep(line_num=s.line_num, command="wait", argument="0.3s", raw_line="/wait 0.3s"))
                new_steps.append(MacroStep(line_num=s.line_num, command="interact", argument="", raw_line="/interact"))
                has_interaction = True

        return Macro(name=existing_macro.name, steps=new_steps, file_path=existing_macro.file_path)

    def _rule_based_formulation(
        self,
        query: str,
        visible_elements: List[str],
        macro_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Deterministic fallback formulation grounded in workspace file context."""
        context = context or self.gather_workspace_context(query)
        lines = [f"# Macro: {macro_name}", f"# Grounded in workspace context for query: '{query}'"]
        q_lower = query.lower()

        # 1. Specialized Coding Task: Modifying platforms in pixi.toml
        if "pixi" in q_lower and ("linux" in q_lower or "platform" in q_lower):
            open_pixi = MACROS_DIR / "open_pixi.macro"
            if open_pixi.exists():
                lines.append('/macro "open_pixi"')
            else:
                lines.append('/openfile "pixi.toml"')
                lines.append('/wait 0.3s')
                lines.append('/interact')
                lines.append('/wait 0.2s')

            # Inspect actual pixi.toml from context
            pixi_info = context.get("target_files", {}).get("pixi.toml")
            target_line = 4
            if pixi_info:
                for idx, line in enumerate(pixi_info["raw_lines"], start=1):
                    if line.strip().startswith("platforms ="):
                        target_line = idx
                        break

            lines.append(f'/gotoline {target_line}')
            lines.append('/press end')
            lines.append('/press left')
            lines.append('/type \', "linux-64"\'')
            lines.append('/press ctrl+s')
            return "\n".join(lines) + "\n"

        # 2. General File Modification Task
        file_match = re.search(r'([a-zA-Z0-9_\-]+\.(?:toml|py|json|md|yaml|js|html|css))', query, re.IGNORECASE)
        if file_match and ("add" in q_lower or "edit" in q_lower or "modify" in q_lower or "update" in q_lower):
            filename = file_match.group(1)
            lines.append(f'/openfile "{filename}"')
            lines.append('/wait 0.3s')
            lines.append('/interact')
            lines.append('/wait 0.2s')

            file_info = context.get("target_files", {}).get(filename)
            target_line = None
            setting_match = re.search(r'(?:setting|property|field|line)\s+["\']?([a-zA-Z0-9_\-]+)["\']?', query, re.IGNORECASE)
            if setting_match and file_info:
                term = setting_match.group(1).lower()
                for idx, line in enumerate(file_info["raw_lines"], start=1):
                    if term in line.lower():
                        target_line = idx
                        break

            if target_line:
                lines.append(f'/gotoline {target_line}')
            elif setting_match:
                lines.append(f'/findtext "{setting_match.group(1)}"')
            lines.append('/press end')
            lines.append('/press ctrl+s')
            return "\n".join(lines) + "\n"

        # 3. Visual Element Target
        target = None
        for elem in sorted(visible_elements, key=len, reverse=True):
            if len(elem) >= 3 and re.search(rf"\b{re.escape(elem.lower())}\b", q_lower):
                target = elem
                break

        if not target:
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
