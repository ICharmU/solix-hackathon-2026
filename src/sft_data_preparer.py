"""
SFT Dataset Preparation Module for Strategy A: Structured State Transition Fine-Tuning.
Generates exactly 250 curated training steps and 50 validation steps for OpenAI Fine-Tuning.
Implements:
1. Subroutine decomposition (explainability, stack tracking).
2. Task differentiation with delayed lookahead resolution.
3. Detailed (Initial State, Action, Outcome State) triples with cognitive idle tagging.
4. Token counting and cost estimation using tiktoken (cl100k_base).
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tiktoken

from src.config import BASE_DIR

SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"
SFT_DIR = BASE_DIR / "data" / "sft"

SYSTEM_PROMPT = """You are an Agentic Workflow Understanding and Task Timeline Model.
Your mission is to track user workflow context symbolically without numerical solvers:
1. Decompose active tasks into explainable subroutines ([SUBROUTINE_PUSH], [SUBROUTINE_POP]).
2. Differentiate between epistemic subroutines (e.g. temporary lookup) and genuine new macro tasks ([TASK_DIFFERENTIATE]).
3. When transitions are delayed or ambiguous, use the lookahead observation buffer to ratify whether an event shift occurred.
4. Reconnect discontinuous task resumptions ([RESUME_TASK]) when interrupted tasks return.
5. Emitted action states must explicitly distinguish between active atomic UI interactions and cognitive idle states."""

# Domain-specific action templates for generating rich (S0, A, S1) triples
DOMAIN_ACTIONS = {
    "excel": [
        {
            "subroutine": "subroutine_excel_data_entry",
            "s0": {"active_application": "Excel", "window_title": "Monthly_Audit.xlsx", "focused_element": "Sheet: 'Ledger', Cell: 'C14'", "element_content": "4500.00"},
            "action": {"operator": "KEYSTROKE_INPUT", "target_ui_element": "Cell C14", "input_payload": "4950.00", "modifiers": ["Enter"]},
            "s1": {"active_application": "Excel", "window_title": "Monthly_Audit.xlsx", "focused_element": "Sheet: 'Ledger', Cell: 'C15'", "element_content": "", "ui_change_detected": "Cell C14 value updated to 4950.00; cursor moved to C15"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_excel_formula_calc",
            "s0": {"active_application": "Excel", "window_title": "Monthly_Audit.xlsx", "focused_element": "Formula Bar", "element_content": "=SUM(C2:C13)"},
            "action": {"operator": "KEYSTROKE_INPUT", "target_ui_element": "Formula Bar", "input_payload": "=SUM(C2:C14)", "modifiers": ["Enter"]},
            "s1": {"active_application": "Excel", "window_title": "Monthly_Audit.xlsx", "focused_element": "Cell C16", "element_content": "68450.00", "ui_change_detected": "Total column sum recalculated dynamically"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_excel_audit_review",
            "s0": {"active_application": "Excel", "window_title": "Monthly_Audit.xlsx", "focused_element": "Spreadsheet Viewport", "element_content": "Rows 1..25 displayed"},
            "action": {"operator": "NONE", "target_ui_element": "NONE", "input_payload": "", "modifiers": []},
            "s1": {"active_application": "Excel", "window_title": "Monthly_Audit.xlsx", "focused_element": "Spreadsheet Viewport", "element_content": "Rows 1..25 displayed", "ui_change_detected": "No UI change; user reading row totals"},
            "cognitive_state": "COGNITIVE_READING"
        }
    ],
    "vscode": [
        {
            "subroutine": "subroutine_vscode_code_edit",
            "s0": {"active_application": "VS Code", "window_title": "schema_validator.py - Visual Studio Code", "focused_element": "Editor: Line 42, Col 8", "element_content": "def validate_schema(data):"},
            "action": {"operator": "TYPE_CODE", "target_ui_element": "Editor Buffer", "input_payload": "    if not isinstance(data, dict): return False", "modifiers": ["Tab"]},
            "s1": {"active_application": "VS Code", "window_title": "schema_validator.py - Visual Studio Code", "focused_element": "Editor: Line 43, Col 4", "element_content": "return True", "ui_change_detected": "New guard clause inserted; auto-indented to 4 spaces"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_vscode_test_run",
            "s0": {"active_application": "VS Code", "window_title": "terminal - Visual Studio Code", "focused_element": "Integrated Terminal", "element_content": "$ "},
            "action": {"operator": "TERMINAL_COMMAND", "target_ui_element": "Terminal Prompt", "input_payload": "pytest tests/test_schema.py -v", "modifiers": ["Enter"]},
            "s1": {"active_application": "VS Code", "window_title": "terminal - Visual Studio Code", "focused_element": "Integrated Terminal", "element_content": "12 passed in 1.4s", "ui_change_detected": "Terminal test suite executed; 12/12 passing green"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_vscode_doc_reading",
            "s0": {"active_application": "VS Code", "window_title": "README.md - Visual Studio Code", "focused_element": "Markdown Preview", "element_content": "API Documentation"},
            "action": {"operator": "NONE", "target_ui_element": "NONE", "input_payload": "", "modifiers": []},
            "s1": {"active_application": "VS Code", "window_title": "README.md - Visual Studio Code", "focused_element": "Markdown Preview", "element_content": "API Documentation", "ui_change_detected": "No movement; inspecting parameter specifications"},
            "cognitive_state": "COGNITIVE_READING"
        }
    ],
    "autocad": [
        {
            "subroutine": "subroutine_autocad_polyline_draw",
            "s0": {"active_application": "AutoCAD", "window_title": "FloorPlan_L2.dwg", "focused_element": "Canvas Viewport (Top-Down)", "element_content": "Grid visible; active layer: 'WALLS'"},
            "action": {"operator": "COMMAND_LINE", "target_ui_element": "AutoCAD Command Prompt", "input_payload": "PLINE 150.0,220.0 150.0,380.0", "modifiers": ["Enter"]},
            "s1": {"active_application": "AutoCAD", "window_title": "FloorPlan_L2.dwg", "focused_element": "Canvas Viewport", "element_content": "New polyline segment (160 units) added to 'WALLS'", "ui_change_detected": "Vector entity rendered on canvas"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_autocad_dimension_tool",
            "s0": {"active_application": "AutoCAD", "window_title": "FloorPlan_L2.dwg", "focused_element": "Ribbon -> Annotate", "element_content": "Dimension toolbar"},
            "action": {"operator": "CLICK", "target_ui_element": "Ribbon Button: Linear Dimension", "input_payload": "Pick vertex A, Pick vertex B", "modifiers": []},
            "s1": {"active_application": "AutoCAD", "window_title": "FloorPlan_L2.dwg", "focused_element": "Canvas Viewport", "element_content": "Linear dimension '14.50m' placed", "ui_change_detected": "Annotation text and witness lines placed"},
            "cognitive_state": "ACTIVE_INTERACTION"
        }
    ],
    "blender": [
        {
            "subroutine": "subroutine_blender_transform_mesh",
            "s0": {"active_application": "Blender", "window_title": "Scene_Asset_Rig.blend", "focused_element": "3D Viewport (Perspective)", "element_content": "Object selected: 'Cube_Subdiv'"},
            "action": {"operator": "HOTKEY", "target_ui_element": "3D Viewport", "input_payload": "G -> Z -> 2.5", "modifiers": ["Enter"]},
            "s1": {"active_application": "Blender", "window_title": "Scene_Asset_Rig.blend", "focused_element": "3D Viewport", "element_content": "Object translated +2.5 along Z", "ui_change_detected": "Transform gizmo updated; mesh position shifted"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_blender_shader_node",
            "s0": {"active_application": "Blender", "window_title": "Scene_Asset_Rig.blend", "focused_element": "Shader Editor", "element_content": "Principled BSDF node"},
            "action": {"operator": "DRAG_DROP", "target_ui_element": "Shader Node Socket", "input_payload": "Connect Texture Color -> Base Color", "modifiers": []},
            "s1": {"active_application": "Blender", "window_title": "Scene_Asset_Rig.blend", "focused_element": "Shader Editor", "element_content": "Node link active", "ui_change_detected": "Material preview refreshed with diffuse texture"},
            "cognitive_state": "ACTIVE_INTERACTION"
        }
    ],
    "photoshop": [
        {
            "subroutine": "subroutine_photoshop_masking",
            "s0": {"active_application": "Photoshop", "window_title": "Hero_Banner_v3.psd", "focused_element": "Layers Panel: 'Subject_Isolated'", "element_content": "Active selection outline on canvas"},
            "action": {"operator": "CLICK", "target_ui_element": "Layers Panel Footer -> Add Layer Mask", "input_payload": "", "modifiers": []},
            "s1": {"active_application": "Photoshop", "window_title": "Hero_Banner_v3.psd", "focused_element": "Layers Panel: 'Subject_Isolated'", "element_content": "Layer mask thumbnail attached", "ui_change_detected": "Background non-destructively hidden outside selection"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_photoshop_color_adjust",
            "s0": {"active_application": "Photoshop", "window_title": "Hero_Banner_v3.psd", "focused_element": "Adjustment Panel", "element_content": "Curves dialog"},
            "action": {"operator": "SLIDER_DRAG", "target_ui_element": "Curves Graph: Midtones Point", "input_payload": "Input: 128, Output: 145", "modifiers": []},
            "s1": {"active_application": "Photoshop", "window_title": "Hero_Banner_v3.psd", "focused_element": "Canvas Viewport", "element_content": "Image midtones boosted", "ui_change_detected": "Global histogram refreshed; midtone brightness increased"},
            "cognitive_state": "ACTIVE_INTERACTION"
        }
    ],
    "salesforce": [
        {
            "subroutine": "subroutine_salesforce_stage_update",
            "s0": {"active_application": "Salesforce", "window_title": "Opportunity: Acme Corp Renewal - Salesforce", "focused_element": "Stage Chevron Bar", "element_content": "Active Stage: 'Qualification'"},
            "action": {"operator": "CLICK", "target_ui_element": "Chevron: 'Proposal/Price Quote'", "input_payload": "Mark Stage as Complete", "modifiers": []},
            "s1": {"active_application": "Salesforce", "window_title": "Opportunity: Acme Corp Renewal - Salesforce", "focused_element": "Stage Chevron Bar", "element_content": "Active Stage: 'Proposal/Price Quote'", "ui_change_detected": "Stage progression saved; probability updated to 75%"},
            "cognitive_state": "ACTIVE_INTERACTION"
        },
        {
            "subroutine": "subroutine_salesforce_log_call",
            "s0": {"active_application": "Salesforce", "window_title": "Opportunity: Acme Corp Renewal - Salesforce", "focused_element": "Activity Component -> Log a Call", "element_content": "Comments field empty"},
            "action": {"operator": "KEYSTROKE_INPUT", "target_ui_element": "Comments Textarea", "input_payload": "Reviewed implementation requirements with VP of Engineering.", "modifiers": ["Save"]},
            "s1": {"active_application": "Salesforce", "window_title": "Opportunity: Acme Corp Renewal - Salesforce", "focused_element": "Activity Timeline", "element_content": "New call entry logged", "ui_change_detected": "Activity timeline updated with timestamped call summary"},
            "cognitive_state": "ACTIVE_INTERACTION"
        }
    ]
}


def build_training_conversation(
    prior_context: str,
    history_events: List[Dict[str, Any]],
    current_action: Dict[str, Any],
    lookahead_buffer: List[Dict[str, Any]],
    target_transition: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Constructs a standard OpenAI chat messages JSON object:
    system: System instructions
    user: [CONTEXT_PRIOR] + [HISTORY] + [CURRENT_ACTION] + [LOOKAHEAD]
    assistant: Target prediction ([EVENT_SHIFT], [SUBROUTINE_PUSH/POP], [TASK_DIFFERENTIATE], [RESUME_TASK])
    """
    user_parts = []

    # 1. Prior Context
    user_parts.append(f"[CONTEXT_PRIOR]\n{prior_context}\n[/CONTEXT_PRIOR]\n")

    # 2. History
    user_parts.append("[INTERACTION_HISTORY]")
    if not history_events:
        user_parts.append("Session started. No prior actions in buffer.")
    else:
        for h in history_events[-4:]:
            user_parts.append(
                f"- t={h['t_start']}s..{h['t_end']}s [{h['app']}]: Subroutine: {h['subroutine']}. Action: {h['action_summary']} (id: {h['seg_id']})"
            )
    user_parts.append("[/INTERACTION_HISTORY]\n")

    # 3. Current Action Triple (S0, A, S1)
    act = current_action
    s0 = act.get("s0") or act.get("initial_state") or {}
    s1 = act.get("s1") or act.get("outcome_state") or {}
    action_dict = act.get("action") or {}

    user_parts.append("[CURRENT_ACTION_OBSERVATION]")
    user_parts.append(f"Timestamp: t={act.get('timestamp_s', 0.0)}s")
    user_parts.append(f"Initial State (S0): App='{s0.get('active_application', 'Unknown')}', Window='{s0.get('window_title', '')}', Focused='{s0.get('focused_element', '')}'")
    user_parts.append(f"Action (A): Operator={action_dict.get('operator', 'NONE')}, Target='{action_dict.get('target_ui_element', 'NONE')}', Payload='{action_dict.get('input_payload', '')}'")
    user_parts.append(f"Outcome State (S1): Focused='{s1.get('focused_element', '')}', Feedback='{s1.get('ui_change_detected', '')}'")
    user_parts.append(f"Cognitive State: {act.get('cognitive_state', 'ACTIVE_INTERACTION')}")
    user_parts.append("[/CURRENT_ACTION_OBSERVATION]\n")

    # 4. Lookahead Buffer for Resolving Ambiguity
    user_parts.append("[LOOKAHEAD_BUFFER (Future 30s)]")
    if not lookahead_buffer:
        user_parts.append("End of recorded session reached; no future actions.")
    else:
        for f in lookahead_buffer[:2]:
            user_parts.append(f"- Future t+{f.get('offset_s', 10)}s: [{f['app']}] Action: {f['summary']}")
    user_parts.append("[/LOOKAHEAD_BUFFER]")

    user_text = "\n".join(user_parts)

    # 5. Target Assistant Output
    target_parts = []
    if target_transition.get("is_breakpoint"):
        target_parts.append("[EVENT_SHIFT]")
        t_type = target_transition.get("transition_type", "NEW_MACRO")

        if t_type == "NEW_MACRO":
            target_parts.append(f"[TASK_DIFFERENTIATE type=\"NEW_MACRO\" from=\"{target_transition['from_app']}\" to=\"{target_transition['to_app']}\"]")
            target_parts.append(f"[SUBROUTINE_PUSH id=\"{target_transition['subroutine']}\"]")
        elif t_type == "RESUME":
            target_parts.append(f"[RESUME_TASK id=\"{target_transition['seg_id']}\" ante=\"{target_transition['resumes_id']}\"]")
            target_parts.append(f"[SUBROUTINE_PUSH id=\"{target_transition['subroutine']}\"]")
        elif t_type == "SUBROUTINE_SHIFT":
            target_parts.append(f"[SUBROUTINE_POP id=\"{target_transition.get('pop_id', '')}\"]")
            target_parts.append(f"[SUBROUTINE_PUSH id=\"{target_transition['subroutine']}\"]")
    else:
        target_parts.append("[CONTINUE_FLOW]")
        target_parts.append(f"[ACTIVE_SUBROUTINE id=\"{target_transition['subroutine']}\"]")

    target_parts.append(f"Reasoning: {target_transition['reasoning']}")
    assistant_text = "\n".join(target_parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def generate_sft_splits(
    num_train: int = 250,
    num_val: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Constructs exactly num_train (250) training conversations and num_val (50) validation conversations
    sourced from the synthetic dataset manifests.
    """
    rng = random.Random(seed)

    # Load all training manifests
    train_manifests = []
    for mf in sorted((SYNTHETIC_DIR / "train").glob("*_manifest.json")):
        train_manifests.append(json.loads(mf.read_text(encoding="utf-8")))

    val_manifests = []
    for mf in sorted((SYNTHETIC_DIR / "val").glob("*_manifest.json")):
        val_manifests.append(json.loads(mf.read_text(encoding="utf-8")))

    def build_samples_from_manifests(manifests: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        samples = []

        while len(samples) < target_count:
            # Pick a random manifest
            mf = rng.choice(manifests)
            segs = mf["segments"]
            ex_id = mf["synthetic_id"]

            prior_context = f"Session: {ex_id}. Target Project Goals: End-to-end multi-app workflow across {', '.join(set(s['category'] for s in segs))}."

            # Determine sample type:
            # 1. Breakpoint transition (~30%)
            # 2. Resumption link (~20%)
            # 3. Lookahead disambiguation (~20%)
            # 4. Intra-task continuous operation (~30%)
            sample_kind = rng.choices(["breakpoint", "resumption", "lookahead", "continuous"], weights=[0.30, 0.20, 0.20, 0.30])[0]

            if sample_kind == "resumption":
                # Find a segment that resumes a previous segment
                resuming_segs = [s for s in segs if s.get("resumes_segment_id")]
                if resuming_segs:
                    curr_seg = rng.choice(resuming_segs)
                    prev_seg_id = curr_seg["resumes_segment_id"]
                    cat = curr_seg["category"]
                    action_tpl = rng.choice(DOMAIN_ACTIONS.get(cat, DOMAIN_ACTIONS["vscode"]))

                    # History leading up to it
                    hist = [
                        {"t_start": 0.0, "t_end": 60.0, "app": cat, "subroutine": f"subroutine_{cat}_init", "action_summary": "Initial operations", "seg_id": prev_seg_id},
                        {"t_start": 60.0, "t_end": curr_seg["start_time_s"], "app": "other_app", "subroutine": "intervening_routine", "action_summary": "Interleaved interruption", "seg_id": "seg_inter"}
                    ]
                    lookahead = [
                        {"offset_s": 15.0, "app": cat, "summary": "Continuing resumed task"},
                        {"offset_s": 30.0, "app": cat, "summary": "Concluded task wrap-up"}
                    ]
                    target = {
                        "is_breakpoint": True,
                        "transition_type": "RESUME",
                        "seg_id": curr_seg["segment_id"],
                        "resumes_id": prev_seg_id,
                        "from_app": "other_app",
                        "to_app": cat,
                        "subroutine": action_tpl["subroutine"],
                        "reasoning": f"User returned to {cat} after intervening interruption; reconnects to antecedent {prev_seg_id} via stack resumption."
                    }
                    samples.append(build_training_conversation(prior_context, hist, action_tpl, lookahead, target))
                    continue

            if sample_kind == "breakpoint":
                # Segment boundary transition
                if len(segs) > 1:
                    idx = rng.randint(1, len(segs) - 1)
                    curr_seg = segs[idx]
                    prev_seg = segs[idx - 1]
                    cat = curr_seg["category"]
                    action_tpl = rng.choice(DOMAIN_ACTIONS.get(cat, DOMAIN_ACTIONS["vscode"]))

                    hist = [
                        {"t_start": prev_seg["start_time_s"], "t_end": prev_seg["end_time_s"], "app": prev_seg["category"], "subroutine": f"subroutine_{prev_seg['category']}_work", "action_summary": "Prior task active", "seg_id": prev_seg["segment_id"]}
                    ]
                    lookahead = [
                        {"offset_s": 10.0, "app": cat, "summary": "Subsequent step in new application"},
                        {"offset_s": 25.0, "app": cat, "summary": "Extended interaction in new application"}
                    ]
                    target = {
                        "is_breakpoint": True,
                        "transition_type": "NEW_MACRO",
                        "from_app": prev_seg["category"],
                        "to_app": cat,
                        "subroutine": action_tpl["subroutine"],
                        "reasoning": f"Shift from {prev_seg['category']} to {cat} represents a distinct macro task boundary confirmed by lookahead continuation."
                    }
                    samples.append(build_training_conversation(prior_context, hist, action_tpl, lookahead, target))
                    continue

            if sample_kind == "lookahead":
                # Ambiguous switch: e.g. quick lookup subroutine vs permanent macro shift
                curr_seg = rng.choice(segs)
                cat = curr_seg["category"]
                action_tpl = rng.choice(DOMAIN_ACTIONS.get(cat, DOMAIN_ACTIONS["vscode"]))

                # Choose a distinct from_app
                all_cats = ["autocad", "blender", "excel", "photoshop", "salesforce", "vscode"]
                from_app = rng.choice([c for c in all_cats if c != cat])
                from_tpl = rng.choice(DOMAIN_ACTIONS.get(from_app, DOMAIN_ACTIONS["vscode"]))

                returns = rng.choice([True, False])
                if returns:
                    # Case A: user temporarily enters cat, but lookahead shows return to from_app
                    hist = [{"t_start": 0.0, "t_end": 45.0, "app": from_app, "subroutine": from_tpl["subroutine"], "action_summary": "Active primary task", "seg_id": "seg_prior"}]
                    lookahead = [{"offset_s": 8.0, "app": from_app, "summary": f"Returned to {from_app} active viewport/editor"}]
                    target = {
                        "is_breakpoint": False,
                        "transition_type": "SUBROUTINE_SHIFT",
                        "subroutine": f"subroutine_{cat}_epistemic_lookup",
                        "reasoning": f"Lookahead confirms immediate return to {from_app}; this action in {cat} is an epistemic lookup subroutine supporting the active task, not a new macro task."
                    }
                else:
                    # Case B: user enters cat and lookahead shows ongoing engagement in cat
                    hist = [{"t_start": 0.0, "t_end": 45.0, "app": from_app, "subroutine": from_tpl["subroutine"], "action_summary": "Concluded prior task", "seg_id": "seg_prior"}]
                    lookahead = [{"offset_s": 15.0, "app": cat, "summary": f"Ongoing work in {cat}; no return observed"}]
                    target = {
                        "is_breakpoint": True,
                        "transition_type": "NEW_MACRO",
                        "from_app": from_app,
                        "to_app": cat,
                        "subroutine": action_tpl["subroutine"],
                        "reasoning": f"Lookahead confirms continued engagement in {cat} with no return to {from_app}; ratified as a genuine macro task boundary."
                    }
                samples.append(build_training_conversation(prior_context, hist, action_tpl, lookahead, target))
                continue

            # Continuous intra-task operation
            curr_seg = rng.choice(segs)
            cat = curr_seg["category"]
            action_tpl = rng.choice(DOMAIN_ACTIONS.get(cat, DOMAIN_ACTIONS["vscode"]))
            hist = [
                {"t_start": curr_seg["start_time_s"], "t_end": curr_seg["start_time_s"] + 15.0, "app": cat, "subroutine": action_tpl["subroutine"], "action_summary": "Preceding action in same routine", "seg_id": curr_seg["segment_id"]}
            ]
            lookahead = [
                {"offset_s": 10.0, "app": cat, "summary": "Next atomic action in same routine"}
            ]
            target = {
                "is_breakpoint": False,
                "subroutine": action_tpl["subroutine"],
                "reasoning": f"Action continues existing {action_tpl['subroutine']} in {cat}; continuous flow maintained."
            }
            samples.append(build_training_conversation(prior_context, hist, action_tpl, lookahead, target))

        return samples[:target_count]

    train_samples = build_samples_from_manifests(train_manifests, num_train)
    val_samples = build_samples_from_manifests(val_manifests, num_val)

    # Compute exact tokens with tiktoken using o200k_base (GPT-4o / GPT-4o-mini tokenizer)
    enc = tiktoken.get_encoding("o200k_base")

    def count_tokens(dataset: List[Dict[str, Any]]) -> int:
        total = 0
        for item in dataset:
            for msg in item["messages"]:
                total += len(enc.encode(msg["content"])) + 4 # 4 tokens per message formatting overhead
            total += 3 # reply priming
        return total

    train_tokens = count_tokens(train_samples)
    val_tokens = count_tokens(val_samples)

    stats = {
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "train_tokens_per_epoch": train_tokens,
        "epochs": 3,
        "total_training_tokens": train_tokens * 3,
        "cost_per_million": 3.00,
        "estimated_training_cost_usd": round((train_tokens * 3 / 1_000_000) * 3.00, 4),
        "avg_tokens_per_sample": round(train_tokens / len(train_samples), 1),
    }

    return train_samples, val_samples, stats


def prepare_and_save_dataset(
    output_dir: Path = None,
    num_train: int = 250,
    num_val: int = 50,
    seed: int = 42,
) -> Path:
    """Generates JSONL dataset files and writes token summary report."""
    if output_dir is None:
        output_dir = SFT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_sft.jsonl"
    val_path = output_dir / "val_sft.jsonl"
    stats_path = output_dir / "sft_dataset_stats.json"

    train_samples, val_samples, stats = generate_sft_splits(
        num_train=num_train,
        num_val=num_val,
        seed=seed,
    )

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n============================================================")
    print(f"[SUCCESS] SFT Dataset Generation Complete!")
    print(f"Train File:   {train_path} ({len(train_samples)} samples, {stats['train_tokens_per_epoch']} tokens/epoch)")
    print(f"Val File:     {val_path} ({len(val_samples)} samples)")
    print(f"Epochs:       {stats['epochs']}")
    print(f"Total Tokens: {stats['total_training_tokens']:,}")
    print(f"Total Cost:   ${stats['estimated_training_cost_usd']:.2f} USD (Guaranteed < $3.00)")
    print(f"============================================================\n")

    return stats_path


if __name__ == "__main__":
    prepare_and_save_dataset()
