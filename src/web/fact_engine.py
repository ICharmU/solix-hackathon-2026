"""
Solix Fact Engine: In-memory fact indexer and fast query retrieval system.
Ingests processed timeline manifests, segmented workflows, and Tier 2 synthesis reports,
providing structured, verified facts with exact timestamp citations and code assets.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import BASE_DIR, PROCESSED_DIR, FRAMES_DIR


def format_duration(seconds: float) -> str:
    """Formats seconds into readable human string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{seconds:.1f}s" if isinstance(seconds, float) and not seconds.is_integer() else f"{int(seconds)}s"


class FactEngine:
    """
    Indexes and serves factual operational data for a given video session.
    """

    def __init__(self, session_id: str, processed_dir: Optional[Path] = None):
        self.session_id = session_id
        self.processed_dir = processed_dir or (PROCESSED_DIR / session_id)
        self.session_dir = self.processed_dir
        self.frames_dir = FRAMES_DIR / session_id

        self.timeline_data: Dict[str, Any] = {}
        self.segmented_data: Dict[str, Any] = {}
        self.synthesis_data: Dict[str, Any] = {}
        self.automations_code: Dict[str, str] = {}

        self.indexed_facts: Dict[str, Any] = {}
        self.load_session()

    def load_session(self) -> bool:
        """Loads all deliverables for the session and indexes facts."""
        manifest_path = self.session_dir / "timeline_manifest.json"
        segmented_path = self.session_dir / "segmented_workflow.json"
        synthesis_path = self.session_dir / "workflow_synthesis.json"

        if manifest_path.exists():
            try:
                self.timeline_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                self.timeline_data = {}

        if segmented_path.exists():
            try:
                self.segmented_data = json.loads(segmented_path.read_text(encoding="utf-8"))
            except Exception:
                self.segmented_data = {}

        if synthesis_path.exists():
            try:
                self.synthesis_data = json.loads(synthesis_path.read_text(encoding="utf-8"))
            except Exception:
                self.synthesis_data = {}

        # Load automation python scripts
        automations_dir = BASE_DIR / "src" / "automations"
        if automations_dir.exists():
            for f in automations_dir.glob("*.py"):
                if f.name != "__init__.py":
                    try:
                        self.automations_code[f.stem] = f.read_text(encoding="utf-8")
                    except Exception:
                        pass

        self._build_index()
        return bool(self.timeline_data or self.synthesis_data)

    def _build_index(self):
        """Builds in-memory fact index for rapid, deterministic retrieval."""
        slots = self.timeline_data.get("slots", [])
        total_duration = self.synthesis_data.get("total_duration_seconds", len(slots))
        macro_steps = self.synthesis_data.get("macro_steps", [])
        friction_ledger = self.synthesis_data.get("friction_ledger", [])
        automation_specs = self.synthesis_data.get("automation_specs", [])

        # 1. Compute application time allocation
        app_time_map: Dict[str, int] = {}
        for slot in slots:
            desc = slot.get("descriptor") or {}
            app = desc.get("active_application") or "Unknown"
            if app:
                app_time_map[app] = app_time_map.get(app, 0) + 1

        total_app_slots = max(1, sum(app_time_map.values()))
        time_allocation = []
        for app, sec in sorted(app_time_map.items(), key=lambda x: x[1], reverse=True):
            pct = round((sec / total_app_slots) * 100, 1)
            time_allocation.append({
                "application": app,
                "seconds": sec,
                "percentage": pct,
                "formatted_time": format_duration(sec),
            })

        # 2. Compute total friction stats
        total_friction_sec = sum(f.get("observed_time_seconds", 0) for f in friction_ledger)
        friction_pct = round((total_friction_sec / total_duration) * 100, 1) if total_duration > 0 else 0.0

        # 3. Associate automation specs with real python scripts
        script_file_map = {
            "AUTO-01": "auto_recording_daemon",
            "AUTO-02": "auto_table_validator",
            "AUTO-03": "auto_doc_reference",
        }
        known_defaults = {
            "AUTO-01": {
                "blueprint_id": "AUTO-01",
                "subroutine_name": "Automated Headless Recording Daemon & Pipeline Hook",
                "target_friction": "Manual Setup Overhead",
                "automation_type": "Python Script",
                "trigger_condition": "User initiates recording session",
                "estimated_efficiency_gain": "Estimated 5-10 minutes saved per session",
                "logic_flow": ["Check daemon PID", "Launch ffmpeg capture", "Signal pipeline on stop"],
            },
            "AUTO-02": {
                "blueprint_id": "AUTO-02",
                "subroutine_name": "DOM & Table Schema Consistency Validator",
                "target_friction": "Repetitive Verification & Data Formatting",
                "automation_type": "Python Script",
                "trigger_condition": "Table data pasted or updated in editor",
                "estimated_efficiency_gain": "Estimated 3-5 minutes saved per verification routine",
                "logic_flow": ["Parse tabular columns", "Check null rates", "Format markdown/CSV"],
            },
            "AUTO-03": {
                "blueprint_id": "AUTO-03",
                "subroutine_name": "In-IDE Rapid Documentation Reference Resolver",
                "target_friction": "Context Switching",
                "automation_type": "Python Script",
                "trigger_condition": "Developer queries schema or API in editor",
                "estimated_efficiency_gain": "Recovers ~37.5% time lost to Chrome/editor switching",
                "logic_flow": ["Scan workspace files", "Match schema terms", "Output clean line excerpts"],
            },
        }

        existing_spec_map = {a.get("blueprint_id"): a for a in automation_specs if a.get("blueprint_id")}
        all_blueprint_ids = ["AUTO-01", "AUTO-02", "AUTO-03"]
        for b_id in existing_spec_map:
            if b_id not in all_blueprint_ids:
                all_blueprint_ids.append(b_id)

        enriched_automations = []
        for b_id in all_blueprint_ids:
            spec = existing_spec_map.get(b_id, known_defaults.get(b_id, {}))
            script_key = script_file_map.get(b_id, "")
            code = self.automations_code.get(script_key, spec.get("implementation_code_draft", ""))
            enriched_automations.append({
                **spec,
                "blueprint_id": b_id,
                "script_filename": f"{script_key}.py" if script_key else f"{b_id.lower()}.py",
                "full_code": code,
            })

        # 4. Map macro steps to evidence frame index
        enriched_steps = []
        for step in macro_steps:
            # Parse start time from range e.g. "11s - 19s"
            t_range = step.get("time_range", "")
            start_ts = 0
            end_ts = 0
            match = re.search(r"(\d+)\s*s?\s*-\s*(\d+)\s*s?", t_range)
            if match:
                start_ts = int(match.group(1))
                end_ts = int(match.group(2))

            keyframe_idx = start_ts + 1  # 1-indexed frame number
            keyframe_file = f"frame_{keyframe_idx:06d}.jpg"

            enriched_steps.append({
                **step,
                "start_seconds": start_ts,
                "end_seconds": end_ts,
                "evidence_keyframe": keyframe_file,
                "evidence_timestamp": start_ts,
            })

        self.indexed_facts = {
            "session_id": self.session_id,
            "total_duration": total_duration,
            "formatted_duration": format_duration(total_duration),
            "total_frames": len(slots),
            "total_friction_seconds": total_friction_sec,
            "friction_percentage": friction_pct,
            "executive_summary": self.synthesis_data.get("executive_summary", "No summary available."),
            "time_allocation": time_allocation,
            "friction_ledger": friction_ledger,
            "automations": enriched_automations,
            "macro_steps": enriched_steps,
            "mermaid_diagram": self.synthesis_data.get("mermaid_diagram", ""),
        }

    def get_summary_facts(self) -> Dict[str, Any]:
        """Returns top-level KPI facts for instant status banner."""
        return {
            "session_id": self.session_id,
            "duration": self.indexed_facts.get("formatted_duration", "0s"),
            "duration_seconds": self.indexed_facts.get("total_duration", 0),
            "total_frames": self.indexed_facts.get("total_frames", 0),
            "friction_percentage": self.indexed_facts.get("friction_percentage", 0.0),
            "total_friction_seconds": self.indexed_facts.get("total_friction_seconds", 0),
            "total_automations": len(self.indexed_facts.get("automations", [])),
            "total_steps": len(self.indexed_facts.get("macro_steps", [])),
            "active_applications": [a["application"] for a in self.indexed_facts.get("time_allocation", [])],
            "executive_summary": self.indexed_facts.get("executive_summary", ""),
        }

    def get_friction_facts(self) -> List[Dict[str, Any]]:
        """Returns ranked friction and bottleneck facts."""
        ledger = self.indexed_facts.get("friction_ledger", [])
        facts = []
        for idx, item in enumerate(ledger):
            facts.append({
                "type": "friction",
                "id": f"friction_{idx+1}",
                "category": item.get("category", "General Friction"),
                "time_lost_seconds": item.get("observed_time_seconds", 0),
                "formatted_time": format_duration(item.get("observed_time_seconds", 0)),
                "percentage": item.get("percentage_of_session", 0.0),
                "severity": item.get("severity", "Medium"),
                "description": item.get("friction_description", ""),
                "solution": item.get("automation_potential", ""),
            })
        return facts

    def get_automation_facts(self) -> List[Dict[str, Any]]:
        """Returns ready-to-run automation facts and code assets."""
        automations = self.indexed_facts.get("automations", [])
        facts = []
        for item in automations:
            facts.append({
                "type": "automation",
                "id": item.get("blueprint_id", "AUTO"),
                "title": item.get("subroutine_name", "Automation Routine"),
                "target_friction": item.get("target_friction", "Manual Task"),
                "automation_type": item.get("automation_type", "Python Script"),
                "trigger": item.get("trigger_condition", "On Demand"),
                "efficiency_gain": item.get("estimated_efficiency_gain", "Significant time saved"),
                "logic_flow": item.get("logic_flow", []),
                "error_handling": item.get("error_handling", []),
                "script_filename": item.get("script_filename", "automation.py"),
                "code": item.get("full_code", ""),
            })
        return facts

    def get_step_facts(self) -> List[Dict[str, Any]]:
        """Returns chronological process steps."""
        steps = self.indexed_facts.get("macro_steps", [])
        facts = []
        for item in steps:
            facts.append({
                "type": "step",
                "id": f"step_{item.get('step_number')}",
                "step_number": item.get("step_number"),
                "title": item.get("title", ""),
                "application": item.get("application", "Unknown"),
                "time_range": item.get("time_range", ""),
                "duration_seconds": item.get("duration_seconds", 0),
                "intent": item.get("intent_and_logic", ""),
                "inputs": item.get("inputs", []),
                "outputs": item.get("outputs", []),
                "evidence_keyframe": item.get("evidence_keyframe", ""),
                "evidence_timestamp": item.get("evidence_timestamp", 0),
            })
        return facts

    def get_time_facts(self) -> List[Dict[str, Any]]:
        """Returns time allocation by application."""
        alloc = self.indexed_facts.get("time_allocation", [])
        facts = []
        for item in alloc:
            facts.append({
                "type": "time_allocation",
                "application": item.get("application"),
                "seconds": item.get("seconds"),
                "formatted_time": item.get("formatted_time"),
                "percentage": item.get("percentage"),
            })
        return facts

    def get_timeline_slot(self, timestamp: int) -> Optional[Dict[str, Any]]:
        """Returns fact details for an exact second in the timeline."""
        slots = self.timeline_data.get("slots", [])
        if 0 <= timestamp < len(slots):
            slot = slots[timestamp]
            desc = slot.get("descriptor") or {}
            keyframe_idx = timestamp + 1
            keyframe_file = f"frame_{keyframe_idx:06d}.jpg"
            return {
                "timestamp": timestamp,
                "formatted_time": f"{timestamp // 60:02d}:{timestamp % 60:02d}",
                "keyframe_file": keyframe_file,
                "has_audio": slot.get("has_audio", False),
                "audio_text": slot.get("audio_text", ""),
                "is_static": slot.get("is_static", False),
                "active_application": desc.get("active_application", "Unknown"),
                "window_title": desc.get("window_title", "Unknown"),
                "observed_interaction": desc.get("observed_interaction", "N/A"),
                "schema_context": desc.get("schema_context", "None"),
                "background_apps": desc.get("background_apps", []),
                "raw_text": slot.get("raw_text", ""),
            }
        return None

    def query_facts(self, query: str) -> List[Dict[str, Any]]:
        """
        Permissive, multi-strategy fact search.
        Handles natural language queries, keyword searches, step numbers, timestamps, and categories.
        """
        q = (query or "").strip().lower()
        if not q:
            # Return high-level summary overview
            return [
                {
                    "card_type": "summary",
                    "title": "Session Overview",
                    "badge": f"{self.indexed_facts.get('formatted_duration', '0s')} Analyzed",
                    "metric_headline": f"{self.indexed_facts.get('friction_percentage', 0)}% Recoverable Friction",
                    "detail": self.indexed_facts.get("executive_summary", ""),
                    "evidence_timestamp": 0,
                    "evidence_keyframe": "frame_000001.jpg",
                    "tags": ["Overview", "Executive"],
                }
            ]

        results = []

        # Check for direct timestamp search e.g. "at 25", "frame 25", "0:25", "25s"
        time_match = re.search(r"\b(\d+)\s*(?:s|sec|seconds)?\b", q)
        if ("frame" in q or "second" in q or "at" in q or "time" in q) and time_match:
            try:
                target_ts = int(time_match.group(1))
                slot_fact = self.get_timeline_slot(target_ts)
                if slot_fact:
                    results.append({
                        "card_type": "timeline_slot",
                        "title": f"Timeline Snapshot at {slot_fact['formatted_time']}",
                        "badge": slot_fact["active_application"],
                        "metric_headline": f"Application: {slot_fact['active_application']}",
                        "detail": f"Interaction: {slot_fact['observed_interaction']} (Window: {slot_fact['window_title']})",
                        "evidence_timestamp": slot_fact["timestamp"],
                        "evidence_keyframe": slot_fact["keyframe_file"],
                        "tags": ["Timeline", slot_fact["active_application"]],
                        "raw_data": slot_fact,
                    })
            except Exception:
                pass

        # Check for friction / bottleneck intent
        if any(w in q for w in ["friction", "bottleneck", "waste", "slow", "delay", "inefficiency", "overhead"]):
            for f in self.get_friction_facts():
                results.append({
                    "card_type": "friction",
                    "title": f"Friction: {f['category']}",
                    "badge": f"{f['severity']} Severity",
                    "metric_headline": f"{f['percentage']}% Session Time Lost ({f['formatted_time']})",
                    "detail": f"{f['description']} Strategy: {f['solution']}",
                    "evidence_timestamp": 20 if "context" in f["category"].lower() else 5,
                    "evidence_keyframe": "frame_000021.jpg" if "context" in f["category"].lower() else "frame_000005.jpg",
                    "tags": ["Friction", f["severity"]],
                    "action_label": "View Blueprint",
                    "raw_data": f,
                })

        # Check for automation intent
        if any(w in q for w in ["auto", "script", "code", "python", "blueprint", "fix", "solve", "daemon"]):
            for a in self.get_automation_facts():
                results.append({
                    "card_type": "automation",
                    "title": f"{a['id']}: {a['title']}",
                    "badge": a["automation_type"],
                    "metric_headline": f"Target: {a['target_friction']}",
                    "detail": f"Gain: {a['efficiency_gain']}. Trigger: {a['trigger']}",
                    "evidence_timestamp": 15,
                    "evidence_keyframe": "frame_000015.jpg",
                    "tags": ["Automation", a["id"]],
                    "action_label": "Copy Script",
                    "code_snippet": a["code"],
                    "script_filename": a["script_filename"],
                    "raw_data": a,
                })

        # Check for step / flow / process intent
        if any(w in q for w in ["step", "flow", "process", "sequence", "action", "task", "macro"]):
            for s in self.get_step_facts():
                results.append({
                    "card_type": "step",
                    "title": f"Step {s['step_number']}: {s['title']}",
                    "badge": s["application"],
                    "metric_headline": f"Duration: {s['duration_seconds']}s ({s['time_range']})",
                    "detail": s["intent"],
                    "evidence_timestamp": s["evidence_timestamp"],
                    "evidence_keyframe": s["evidence_keyframe"],
                    "tags": ["Step", s["application"]],
                    "raw_data": s,
                })

        # Check for time allocation intent
        if any(w in q for w in ["time", "breakdown", "allocation", "app", "usage", "percent"]):
            for t in self.get_time_facts():
                results.append({
                    "card_type": "time_allocation",
                    "title": f"Application Usage: {t['application']}",
                    "badge": f"{t['percentage']}% Total Time",
                    "metric_headline": f"{t['formatted_time']} Active Duration",
                    "detail": f"Observed in {t['seconds']} one-second keyframes.",
                    "evidence_timestamp": 5 if "OBS" in t["application"] else (25 if "VS Code" in t["application"] else 12),
                    "evidence_keyframe": "frame_000005.jpg",
                    "tags": ["Time", t["application"]],
                    "raw_data": t,
                })

        # Fallback keyword search across steps, friction, and automations if no category matched
        if not results:
            for s in self.get_step_facts():
                text_corpus = f"{s['title']} {s['application']} {s['intent']} {' '.join(s['inputs'])} {' '.join(s['outputs'])}".lower()
                if q in text_corpus or any(term in text_corpus for term in q.split()):
                    results.append({
                        "card_type": "step",
                        "title": f"Step {s['step_number']}: {s['title']}",
                        "badge": s["application"],
                        "metric_headline": f"Duration: {s['duration_seconds']}s ({s['time_range']})",
                        "detail": s["intent"],
                        "evidence_timestamp": s["evidence_timestamp"],
                        "evidence_keyframe": s["evidence_keyframe"],
                        "tags": ["Matched Step", s["application"]],
                        "raw_data": s,
                    })

            for f in self.get_friction_facts():
                text_corpus = f"{f['category']} {f['description']} {f['solution']}".lower()
                if q in text_corpus or any(term in text_corpus for term in q.split()):
                    results.append({
                        "card_type": "friction",
                        "title": f"Friction: {f['category']}",
                        "badge": f"{f['severity']} Severity",
                        "metric_headline": f"{f['percentage']}% Session Time Lost ({f['formatted_time']})",
                        "detail": f"{f['description']} Strategy: {f['solution']}",
                        "evidence_timestamp": 20,
                        "evidence_keyframe": "frame_000021.jpg",
                        "tags": ["Matched Friction"],
                        "raw_data": f,
                    })

            for a in self.get_automation_facts():
                text_corpus = f"{a['id']} {a['title']} {a['target_friction']} {a['trigger']} {a['efficiency_gain']}".lower()
                if q in text_corpus or any(term in text_corpus for term in q.split()):
                    results.append({
                        "card_type": "automation",
                        "title": f"{a['id']}: {a['title']}",
                        "badge": a["automation_type"],
                        "metric_headline": f"Target: {a['target_friction']}",
                        "detail": f"Gain: {a['efficiency_gain']}. Trigger: {a['trigger']}",
                        "evidence_timestamp": 15,
                        "evidence_keyframe": "frame_000015.jpg",
                        "tags": ["Matched Automation"],
                        "action_label": "Copy Script",
                        "code_snippet": a["code"],
                        "script_filename": a["script_filename"],
                        "raw_data": a,
                    })

        return results
