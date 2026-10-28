"""
Document Context Engine & Contextualized Reclassification Pipeline

This module creates realistic, upfront Project Context Dossiers (PRD goals, active Jira/Linear tickets,
database schemas, and domain constraints) for all training, validation, and test video sessions.
It then reruns the semantic classification of all segments, breakpoints, and state triples
under this new document context schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("document_context_engine")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = WORKSPACE_ROOT / "data" / "synthetic"
DOSSIERS_DIR = WORKSPACE_ROOT / "data" / "context_dossiers"


# --- Schemas ---

class TicketSpecification(BaseModel):
    ticket_id: str
    category: str
    title: str
    target_artifacts: List[str]
    acceptance_criteria: str
    priority: str = "High"


class DocumentContext(BaseModel):
    project_id: str
    project_title: str
    prd_summary: str
    business_unit: str
    active_tickets: List[TicketSpecification]
    data_schemas: Dict[str, Any]
    domain_constraints: List[str]


class ContextualizedSegment(BaseModel):
    segment_id: str
    segment_index: int
    category: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    governing_ticket_id: str
    governing_ticket_title: str
    contextual_role: str  # PRIMARY_OBJECTIVE, EPISTEMIC_SUBROUTINE, CROSS_TASK_INTERRUPTION, TASK_RESUMPTION
    is_interruption: bool
    resumes_segment_id: Optional[str] = None
    resumed_in_segment_id: Optional[str] = None
    subroutine: str
    initial_state_s0: Dict[str, Any]
    action_a: Dict[str, Any]
    outcome_state_s1: Dict[str, Any]
    cognitive_state: str


class ContextualizedBreakpoint(BaseModel):
    time_s: float
    from_segment: str
    to_segment: str
    from_ticket: str
    to_ticket: str
    transition_type: str  # NEW_MACRO, SUBROUTINE_SHIFT, RESUME
    context_rationale: str


class ContextualizedManifest(BaseModel):
    synthetic_id: str
    split: str
    total_duration_seconds: float
    document_context: DocumentContext
    segments: List[ContextualizedSegment]
    breakpoints: List[ContextualizedBreakpoint]


# --- Domain Context Generators ---

CATEGORY_TICKET_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "vscode": {
        "title": "Validate Schema Models & Implement Unit Tests",
        "target_artifacts": ["src/schemas.py", "tests/test_schema.py", "terminal"],
        "criteria": "Guard clauses implemented, pytest passes 100% green without regressions.",
        "subroutines": [
            {
                "subroutine": "subroutine_vscode_schema_edit",
                "s0": {"app": "VS Code", "window": "schemas.py - Visual Studio Code", "focused": "Editor: Line 42", "content": "class DocumentContext(BaseModel):"},
                "action": {"operator": "KEYSTROKE_INPUT", "target": "Editor Line 43", "payload": "    project_id: str = Field(...)", "modifiers": ["Enter"]},
                "s1": {"app": "VS Code", "window": "schemas.py - Visual Studio Code", "focused": "Editor: Line 44", "content": "    project_id: str = Field(...)", "feedback": "Schema attribute appended, syntax valid"},
                "cognitive": "ACTIVE_INTERACTION"
            },
            {
                "subroutine": "subroutine_vscode_test_run",
                "s0": {"app": "VS Code", "window": "terminal - Visual Studio Code", "focused": "Integrated Terminal Prompt", "content": "$ "},
                "action": {"operator": "TERMINAL_COMMAND", "target": "Integrated Terminal", "payload": "pytest tests/test_schema.py -v", "modifiers": ["Enter"]},
                "s1": {"app": "VS Code", "window": "terminal - Visual Studio Code", "focused": "Integrated Terminal Output", "content": "15 passed in 0.8s", "feedback": "All schema tests passing"},
                "cognitive": "ACTIVE_INTERACTION"
            }
        ]
    },
    "autocad": {
        "title": "Draft Structural Wall Polylines & Calibrate Dimension Lines",
        "target_artifacts": ["FloorPlan_L2.dwg", "WALLS layer", "Annotate Ribbon"],
        "criteria": "Outer perimeter polylines enclosed, dimension tolerance under 0.05m.",
        "subroutines": [
            {
                "subroutine": "subroutine_autocad_polyline_draw",
                "s0": {"app": "AutoCAD", "window": "FloorPlan_L2.dwg - AutoCAD", "focused": "Canvas Viewport", "content": "Grid Active; Layer: WALLS"},
                "action": {"operator": "COMMAND_LINE", "target": "AutoCAD Command Prompt", "payload": "PLINE 0,0 250,0 250,180 0,180 C", "modifiers": ["Enter"]},
                "s1": {"app": "AutoCAD", "window": "FloorPlan_L2.dwg - AutoCAD", "focused": "Canvas Viewport", "content": "Closed 2D Polyline created on WALLS layer", "feedback": "Perimeter geometry rendered"},
                "cognitive": "ACTIVE_INTERACTION"
            },
            {
                "subroutine": "subroutine_autocad_dimension_verify",
                "s0": {"app": "AutoCAD", "window": "FloorPlan_L2.dwg - AutoCAD", "focused": "Ribbon -> Dimension", "content": "Linear Dimension tool"},
                "action": {"operator": "CLICK", "target": "Ribbon Button: Linear Dimension", "payload": "Pick vertex 0,0 and 250,0", "modifiers": []},
                "s1": {"app": "AutoCAD", "window": "FloorPlan_L2.dwg - AutoCAD", "focused": "Canvas Viewport", "content": "Dimension callout 25.00m placed", "feedback": "Clearance confirmed"},
                "cognitive": "ACTIVE_INTERACTION"
            }
        ]
    },
    "blender": {
        "title": "Rig 3D Scene Geometry & Connect PBR Material Shaders",
        "target_artifacts": ["Scene_Asset_Rig.blend", "Principled BSDF", "3D Viewport"],
        "criteria": "Mesh normals unified, base color texture linked, transforms normalized.",
        "subroutines": [
            {
                "subroutine": "subroutine_blender_transform_mesh",
                "s0": {"app": "Blender", "window": "Scene_Asset_Rig.blend - Blender", "focused": "3D Viewport", "content": "Selected: Structural_Shell"},
                "action": {"operator": "HOTKEY", "target": "3D Viewport", "payload": "S -> Z -> 1.5", "modifiers": ["Enter"]},
                "s1": {"app": "Blender", "window": "Scene_Asset_Rig.blend - Blender", "focused": "3D Viewport", "content": "Object scaled 1.5 on Z axis", "feedback": "Gizmo updated, mesh geometry refreshed"},
                "cognitive": "ACTIVE_INTERACTION"
            },
            {
                "subroutine": "subroutine_blender_shader_link",
                "s0": {"app": "Blender", "window": "Scene_Asset_Rig.blend - Blender", "focused": "Shader Editor", "content": "Principled BSDF & Image Texture nodes"},
                "action": {"operator": "DRAG_DROP", "target": "Shader Socket: Color", "payload": "Connect to Base Color", "modifiers": []},
                "s1": {"app": "Blender", "window": "Scene_Asset_Rig.blend - Blender", "focused": "Shader Editor", "content": "Noodle link established", "feedback": "Material preview rendered with diffuse texture"},
                "cognitive": "ACTIVE_INTERACTION"
            }
        ]
    },
    "salesforce": {
        "title": "Advance Enterprise Deal Stage & Log Technical Scope",
        "target_artifacts": ["Opportunity: Enterprise Tenant Expansion", "Chevron Stage Bar", "Activity Timeline"],
        "criteria": "Deal moved to Proposal/Review stage, call notes and engineering deliverables logged.",
        "subroutines": [
            {
                "subroutine": "subroutine_salesforce_stage_advance",
                "s0": {"app": "Salesforce", "window": "Opportunity: Enterprise Expansion - Salesforce", "focused": "Stage Bar: Qualification", "content": "Active Stage: Qualification"},
                "action": {"operator": "CLICK", "target": "Chevron: Proposal / Price Quote", "payload": "Mark Stage as Current", "modifiers": []},
                "s1": {"app": "Salesforce", "window": "Opportunity: Enterprise Expansion - Salesforce", "focused": "Stage Bar: Proposal", "content": "Stage updated to Proposal (75% probability)", "feedback": "Opportunity record persisted"},
                "cognitive": "ACTIVE_INTERACTION"
            },
            {
                "subroutine": "subroutine_salesforce_log_activity",
                "s0": {"app": "Salesforce", "window": "Opportunity: Enterprise Expansion - Salesforce", "focused": "Activity Component -> Log Call", "content": "Comments field empty"},
                "action": {"operator": "KEYSTROKE_INPUT", "target": "Comments Field", "payload": "Completed architectural review with facilities team. Proposal sent.", "modifiers": ["Save"]},
                "s1": {"app": "Salesforce", "window": "Opportunity: Enterprise Expansion - Salesforce", "focused": "Activity Timeline", "content": "Call logged by current user", "feedback": "New activity item placed on timeline"},
                "cognitive": "ACTIVE_INTERACTION"
            }
        ]
    },
    "excel": {
        "title": "Compute Cost Variance Ledger & Reconcile Material Bids",
        "target_artifacts": ["Cost_Estimation_Master.xlsx", "Sheet: Materials_Cost", "Formula Bar"],
        "criteria": "Variance column formulas compute without circular references, total budget balanced.",
        "subroutines": [
            {
                "subroutine": "subroutine_excel_formula_calc",
                "s0": {"app": "Excel", "window": "Cost_Estimation_Master.xlsx - Excel", "focused": "Cell E24", "content": "=SUM(E4:E23)"},
                "action": {"operator": "KEYSTROKE_INPUT", "target": "Cell F24", "payload": "=E24-D24", "modifiers": ["Enter"]},
                "s1": {"app": "Excel", "window": "Cost_Estimation_Master.xlsx - Excel", "focused": "Cell F25", "content": "$12,450.00", "feedback": "Variance calculation updated"},
                "cognitive": "ACTIVE_INTERACTION"
            },
            {
                "subroutine": "subroutine_excel_filter_audit",
                "s0": {"app": "Excel", "window": "Cost_Estimation_Master.xlsx - Excel", "focused": "Table Header: Status", "content": "Filter dropdown"},
                "action": {"operator": "CLICK", "target": "Dropdown Filter: Status", "payload": "Select 'Pending Approval'", "modifiers": []},
                "s1": {"app": "Excel", "window": "Cost_Estimation_Master.xlsx - Excel", "focused": "Table View", "content": "14 rows displayed", "feedback": "Filter applied; viewing pending bids"},
                "cognitive": "ACTIVE_INTERACTION"
            }
        ]
    },
    "photoshop": {
        "title": "Grade Architectural Renders & Mask Foreground Subjects",
        "target_artifacts": ["Hero_Render_Composite.psd", "Layers Panel", "Curves Adjustment"],
        "criteria": "Layer mask created without clipping highlights, midtones graded for client deck.",
        "subroutines": [
            {
                "subroutine": "subroutine_photoshop_mask_apply",
                "s0": {"app": "Photoshop", "window": "Hero_Render_Composite.psd - Photoshop", "focused": "Layers Panel: Foreground_Rig", "content": "Active selection border on canvas"},
                "action": {"operator": "CLICK", "target": "Layers Panel -> Add Layer Mask", "payload": "", "modifiers": []},
                "s1": {"app": "Photoshop", "window": "Hero_Render_Composite.psd - Photoshop", "focused": "Layers Panel: Foreground_Rig", "content": "Mask icon attached", "feedback": "Pixels non-destructively masked"},
                "cognitive": "ACTIVE_INTERACTION"
            },
            {
                "subroutine": "subroutine_photoshop_curves_grade",
                "s0": {"app": "Photoshop", "window": "Hero_Render_Composite.psd - Photoshop", "focused": "Adjustment Panel -> Curves", "content": "Curves linear baseline"},
                "action": {"operator": "SLIDER_DRAG", "target": "Curves Anchor (128, 128)", "payload": "Drag to (128, 142)", "modifiers": []},
                "s1": {"app": "Photoshop", "window": "Hero_Render_Composite.psd - Photoshop", "focused": "Canvas Viewport", "content": "Canvas brightness updated", "feedback": "Render contrast heightened for export"},
                "cognitive": "ACTIVE_INTERACTION"
            }
        ]
    }
}


def build_project_context_for_manifest(manifest: Dict[str, Any]) -> DocumentContext:
    """
    Synthesizes a cohesive, professional DocumentContext tailored to the categories
    present in the composite video manifest.
    """
    synth_id = manifest["synthetic_id"]
    categories = sorted(list(set(s["category"] for s in manifest["segments"])))
    split = manifest.get("split", "train")

    cat_names = [c.capitalize() for c in categories]
    project_title = f"{' & '.join(cat_names)} Enterprise Operational Workflow"
    prd_summary = (
        f"Cross-functional operational pipeline executing sequential and interleaved tasks across "
        f"{', '.join(cat_names)}. Requires strict state continuity, rigorous verification of intermediate deliverables, "
        f"and accurate tracking of task interruptions and resumptions."
    )

    active_tickets: List[TicketSpecification] = []
    data_schemas: Dict[str, Any] = {}
    domain_constraints: List[str] = [
        "All intermediate work-in-progress must be saved before switching application contexts.",
        "When an interruption occurs, the primary task state must be preserved and cleanly resumed.",
        "Final outputs must be validated against ticket acceptance criteria."
    ]

    for idx, cat in enumerate(categories, start=1):
        tpl = CATEGORY_TICKET_TEMPLATES.get(cat, CATEGORY_TICKET_TEMPLATES["vscode"])
        t_id = f"TICKET-{cat.upper()[:4]}-{idx:02d}"
        active_tickets.append(
            TicketSpecification(
                ticket_id=t_id,
                category=cat,
                title=f"{cat.capitalize()}: {tpl['title']}",
                target_artifacts=tpl["target_artifacts"],
                acceptance_criteria=tpl["criteria"],
                priority="Critical" if idx == 1 else "High"
            )
        )
        data_schemas[cat] = {
            "primary_artifacts": tpl["target_artifacts"],
            "validation_rule": tpl["criteria"]
        }

    return DocumentContext(
        project_id=f"PRJ-{synth_id.upper()}",
        project_title=project_title,
        prd_summary=prd_summary,
        business_unit="Digital Operations & Technical Design",
        active_tickets=active_tickets,
        data_schemas=data_schemas,
        domain_constraints=domain_constraints
    )


def reclassify_video_under_context(
    manifest_path: Path,
    output_context_path: Path
) -> ContextualizedManifest:
    """
    Loads an existing synthetic manifest, builds its upfront DocumentContext,
    and reclassifies every segment, action triple, and breakpoint under the new schema.
    """
    raw_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    doc_context = build_project_context_for_manifest(raw_data)

    # Build mapping from category to ticket
    cat_to_ticket: Dict[str, TicketSpecification] = {
        t.category: t for t in doc_context.active_tickets
    }

    raw_segs = raw_data["segments"]
    context_segs: List[ContextualizedSegment] = []

    for seg in raw_segs:
        cat = seg["category"]
        ticket = cat_to_ticket.get(cat, doc_context.active_tickets[0])
        tpl = CATEGORY_TICKET_TEMPLATES.get(cat, CATEGORY_TICKET_TEMPLATES["vscode"])

        # Determine contextual role
        if seg.get("resumes_segment_id"):
            role = "TASK_RESUMPTION"
        elif seg.get("is_interruption"):
            role = "CROSS_TASK_INTERRUPTION"
        elif seg["segment_index"] == 1:
            role = "PRIMARY_OBJECTIVE"
        else:
            role = "IN_CONTEXT_SUBROUTINE"

        # Choose subroutine and triple
        sub_list = tpl["subroutines"]
        sub_choice = sub_list[seg["segment_index"] % len(sub_list)]

        context_segs.append(
            ContextualizedSegment(
                segment_id=seg["segment_id"],
                segment_index=seg["segment_index"],
                category=cat,
                start_time_s=seg["start_time_s"],
                end_time_s=seg["end_time_s"],
                duration_s=seg["duration_s"],
                governing_ticket_id=ticket.ticket_id,
                governing_ticket_title=ticket.title,
                contextual_role=role,
                is_interruption=seg.get("is_interruption", False),
                resumes_segment_id=seg.get("resumes_segment_id"),
                resumed_in_segment_id=seg.get("resumed_in_segment_id"),
                subroutine=sub_choice["subroutine"],
                initial_state_s0=sub_choice["s0"],
                action_a=sub_choice["action"],
                outcome_state_s1=sub_choice["s1"],
                cognitive_state=sub_choice["cognitive"]
            )
        )

    # Reclassify breakpoints
    context_bps: List[ContextualizedBreakpoint] = []
    for i in range(len(context_segs) - 1):
        cur_seg = context_segs[i]
        nxt_seg = context_segs[i + 1]
        bp_time = cur_seg.end_time_s

        if nxt_seg.resumes_segment_id:
            t_type = "RESUME"
            rationale = (
                f"Resuming {nxt_seg.governing_ticket_id} ({nxt_seg.governing_ticket_title}) "
                f"which was previously interrupted by {cur_seg.governing_ticket_id}."
            )
        elif cur_seg.governing_ticket_id != nxt_seg.governing_ticket_id:
            t_type = "NEW_MACRO"
            rationale = (
                f"Transitioning from ticket {cur_seg.governing_ticket_id} to new objective "
                f"{nxt_seg.governing_ticket_id}."
            )
        else:
            t_type = "SUBROUTINE_SHIFT"
            rationale = (
                f"Sequential progression within {cur_seg.governing_ticket_id}: "
                f"{cur_seg.subroutine} -> {nxt_seg.subroutine}."
            )

        context_bps.append(
            ContextualizedBreakpoint(
                time_s=bp_time,
                from_segment=cur_seg.segment_id,
                to_segment=nxt_seg.segment_id,
                from_ticket=cur_seg.governing_ticket_id,
                to_ticket=nxt_seg.governing_ticket_id,
                transition_type=t_type,
                context_rationale=rationale
            )
        )

    c_manifest = ContextualizedManifest(
        synthetic_id=raw_data["synthetic_id"],
        split=raw_data.get("split", "train"),
        total_duration_seconds=raw_data["total_duration_seconds"],
        document_context=doc_context,
        segments=context_segs,
        breakpoints=context_bps
    )

    # Save contextualized manifest
    output_context_path.write_text(json.dumps(c_manifest.model_dump(), indent=2), encoding="utf-8")
    return c_manifest


def process_all_datasets() -> Dict[str, Any]:
    """
    Iterates through all train, val, and test synthetic manifests, creates their document context dossiers,
    and writes the reclassified contextual manifests.
    """
    DOSSIERS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "processed_splits": {},
        "total_videos_reclassified": 0,
        "total_segments_contextualized": 0,
        "total_breakpoints_reclassified": 0
    }

    for split in ["train", "val", "test"]:
        split_dir = SYNTHETIC_DIR / split
        if not split_dir.exists():
            continue

        manifest_files = sorted(split_dir.glob("synthetic_*_manifest.json"))
        # Exclude already contextualized files if any
        manifest_files = [f for f in manifest_files if not f.name.endswith("_context_manifest.json")]

        split_summary = {"videos": len(manifest_files), "items": []}

        for mf in manifest_files:
            clean_stem = mf.stem.replace("_manifest", "").replace("_context", "")
            c_name = f"{clean_stem}_context_manifest.json"
            c_path = split_dir / c_name
            c_manifest = reclassify_video_under_context(mf, c_path)

            # Also save dossier independently in data/context_dossiers/
            dossier_path = DOSSIERS_DIR / f"{c_manifest.synthetic_id}_dossier.json"
            dossier_path.write_text(
                json.dumps(c_manifest.document_context.model_dump(), indent=2),
                encoding="utf-8"
            )

            split_summary["items"].append({
                "synthetic_id": c_manifest.synthetic_id,
                "project_title": c_manifest.document_context.project_title,
                "tickets_count": len(c_manifest.document_context.active_tickets),
                "segments": len(c_manifest.segments),
                "breakpoints": len(c_manifest.breakpoints)
            })

            summary["total_videos_reclassified"] += 1
            summary["total_segments_contextualized"] += len(c_manifest.segments)
            summary["total_breakpoints_reclassified"] += len(c_manifest.breakpoints)

        summary["processed_splits"][split] = split_summary

    # Write master summary
    (DOSSIERS_DIR / "context_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Successfully contextualized {summary['total_videos_reclassified']} videos across all splits.")
    return summary


if __name__ == "__main__":
    summary = process_all_datasets()
    print(json.dumps(summary, indent=2))

