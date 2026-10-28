"""
Tier 2 Workflow Synthesis Engine.
Uses gpt-4o-mini to infer latent business logic, construct time-allocation & friction ledgers,
generate Mermaid process flowcharts, and produce production-ready automation specifications.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from src.schemas import (
    SegmentedWorkflowManifest,
    WorkflowSynthesisReport,
    MacroWorkflowStep,
    FrictionItem,
    AutomationSpec,
)

load_dotenv()


SYNTHESIS_SYSTEM_PROMPT = """You are an expert Enterprise Process Engineer and AI Systems Architect.
Your task is to analyze reconstructed multimodal workflow traces from a worker's desktop recording session.
The data contains segmented timeline steps, subroutine clusters, and stitched discontinuous sequences.

Synthesize this temporal evidence into an authoritative, actionable Tier 2 Business Logic & Automation Report.
You must return a valid JSON object matching the following structure:
{
  "executive_summary": "Comprehensive 2-3 paragraph synthesis of the observed business workflow, objectives, and findings.",
  "macro_steps": [
    {
      "step_number": 1,
      "title": "Clear Macro Step Name",
      "application": "Active Application",
      "time_range": "0s - 11s",
      "duration_seconds": 11,
      "intent_and_logic": "Why the user performed this action and the underlying business logic",
      "inputs": ["input documents, tabs, or settings observed"],
      "outputs": ["artifacts created, modified, or validated"]
    }
  ],
  "friction_ledger": [
    {
      "category": "e.g. Manual Setup Overhead, Context Switching, Repetitive Verification",
      "observed_time_seconds": 12,
      "percentage_of_session": 21.4,
      "severity": "Low | Medium | High",
      "friction_description": "Specific friction, bottleneck, or repetitive task identified",
      "automation_potential": "How this bottleneck can be eliminated or automated"
    }
  ],
  "automation_specs": [
    {
      "blueprint_id": "AUTO-01",
      "subroutine_name": "Descriptive Automation Name",
      "target_friction": "Corresponding friction category",
      "automation_type": "Python Script | Playwright RPA | Hotkey / CLI Hook | API Integration",
      "trigger_condition": "Explicit event or condition triggering this automation",
      "logic_flow": ["Step 1", "Step 2", "Step 3"],
      "error_handling": ["Fallback 1", "Timeout handler", "Exception policy"],
      "estimated_efficiency_gain": "Quantified time or effort savings",
      "implementation_code_draft": "Complete executable Python or script snippet implementing the core logic"
    }
  ],
  "mermaid_diagram": "flowchart TD\\n  A[Start: OBS Studio Setup] --> B[VS Code: Implementation Plan Review]\\n  ..."
}

Ensure the Mermaid diagram is syntactically valid with quoted labels where necessary and proper node connections.
"""


def generate_mock_synthesis(manifest: SegmentedWorkflowManifest) -> WorkflowSynthesisReport:
    """Generates a deterministic synthetic report for testing without API calls."""
    total_duration = sum(s.duration_seconds for s in manifest.segments) if manifest.segments else 56.0

    steps = [
        MacroWorkflowStep(
            step_number=1,
            title="Session Recording Initialization & Setup",
            application="OBS Studio",
            time_range="0s - 10s",
            duration_seconds=11,
            intent_and_logic="Verify desktop audio mixer levels, configure scene transitions, and initialize video recording.",
            inputs=["OBS Studio Desktop Audio Mixer", "Recording Scene Configuration"],
            outputs=["Active Session Recording Stream"],
        ),
        MacroWorkflowStep(
            step_number=2,
            title="Implementation Plan Review & Editing",
            application="VS Code",
            time_range="11s - 31s",
            duration_seconds=21,
            intent_and_logic="Review architectural plan, inspect phase milestones, and edit markdown task specifications.",
            inputs=["implementation_plan.md", "Editor Workspace"],
            outputs=["Updated Task Milestones and Implementation Roadmap"],
        ),
        MacroWorkflowStep(
            step_number=3,
            title="Code Inspection & Verification",
            application="VS Code",
            time_range="32s - 36s",
            duration_seconds=5,
            intent_and_logic="Inspect python source modules and verify data model implementations.",
            inputs=["Python Source Files"],
            outputs=["Code Review Validation"],
        ),
        MacroWorkflowStep(
            step_number=4,
            title="Web Documentation Cross-Reference",
            application="Google Chrome",
            time_range="37s - 41s",
            duration_seconds=5,
            intent_and_logic="Cross-reference API specifications and inspect tabular reference data in browser.",
            inputs=["Web Browser Documentation Tab"],
            outputs=["Verified API Schema Parameters"],
        ),
        MacroWorkflowStep(
            step_number=5,
            title="Interactive Testing & Browser Verification",
            application="Google Chrome",
            time_range="43s - 54s",
            duration_seconds=12,
            intent_and_logic="Navigate web application tabs, inspect table contents, and validate runtime results.",
            inputs=["Chrome Application Dashboard", "Table Contents"],
            outputs=["End-to-End Visual Verification"],
        ),
        MacroWorkflowStep(
            step_number=6,
            title="Session Termination & Teardown",
            application="OBS Studio",
            time_range="55s - 55s",
            duration_seconds=1,
            intent_and_logic="Return to OBS Studio interface and stop the recording.",
            inputs=["OBS Studio Stop Recording Control"],
            outputs=["Final Saved MP4 Video File"],
        ),
    ]

    frictions = [
        FrictionItem(
            category="Setup & Teardown Overhead",
            observed_time_seconds=12,
            percentage_of_session=round(12 / total_duration * 100, 1),
            severity="Medium",
            friction_description="Manual launch, configuration, and shutdown of OBS Studio for recording tracking.",
            automation_potential="Automate capture daemon with system hotkey or CLI wrapper script.",
        ),
        FrictionItem(
            category="Cross-Application Context Switching",
            observed_time_seconds=10,
            percentage_of_session=round(10 / total_duration * 100, 1),
            severity="Medium",
            friction_description="Frequent toggling between VS Code editor and Google Chrome for documentation reference.",
            automation_potential="Embed web documentation search and API references directly into IDE side pane.",
        ),
        FrictionItem(
            category="Manual Documentation Review",
            observed_time_seconds=21,
            percentage_of_session=round(21 / total_duration * 100, 1),
            severity="Low",
            friction_description="Manual scrolling and visual review of markdown implementation plan content.",
            automation_potential="Automate milestone status checks using CI/CD test hooks.",
        ),
    ]

    automation_specs = [
        AutomationSpec(
            blueprint_id="AUTO-01",
            subroutine_name="Automated Screen/Audio Capture CLI",
            target_friction="Setup & Teardown Overhead",
            automation_type="Python Script / OS Hook",
            trigger_condition="Command-line task initiation or hotkey press (e.g. Ctrl+Shift+R)",
            logic_flow=[
                "Invoke headless ffmpeg capture process for desktop display and audio track",
                "Log session start timestamp and write process PID to lockfile",
                "On SIGINT or stop hotkey, terminate capture cleanly and trigger preprocessing pipeline",
            ],
            error_handling=[
                "Check for existing lockfile to prevent duplicate recording instances",
                "Fallback to standard display resolution if virtual monitor resolution is unrecognized",
            ],
            estimated_efficiency_gain="Eliminates ~12s per recording session and removes manual OBS setup friction.",
            implementation_code_draft="""import subprocess, signal, time

def start_capture(output_path="session.mp4"):
    cmd = [
        "ffmpeg", "-y", "-f", "gdigrab", "-framerate", "30",
        "-i", "desktop", "-f", "dshow", "-i", "audio=Stereo Mix",
        "-c:v", "libx264", "-preset", "ultrafast", output_path
    ]
    proc = subprocess.Popen(cmd)
    return proc

def stop_capture(proc):
    proc.send_signal(signal.SIGINT)
    proc.wait()
""",
        ),
        AutomationSpec(
            blueprint_id="AUTO-02",
            subroutine_name="Automated Browser Tab & Table Validator",
            target_friction="Cross-Application Context Switching",
            automation_type="Playwright RPA Script",
            trigger_condition="Post-commit hook or implementation plan milestone change",
            logic_flow=[
                "Launch headless Chromium browser via Playwright",
                "Navigate to target documentation or dashboard URL",
                "Extract DOM table rows and compare schema headers against Pydantic model",
                "Output mismatch report to terminal without requiring manual browser inspection",
            ],
            error_handling=[
                "Retry navigation with 5s exponential backoff on HTTP 5xx errors",
                "Log missing DOM selectors with screenshot artifact on validation failure",
            ],
            estimated_efficiency_gain="Saves ~17s per verification cycle and provides instant programmatic validation.",
            implementation_code_draft="""from playwright.sync_api import sync_playwright

def validate_docs_table(url, expected_columns):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        headers = [th.inner_text().strip() for th in page.query_selector_all("table th")]
        diff = set(expected_columns) - set(headers)
        browser.close()
        return len(diff) == 0, diff
""",
        ),
    ]

    mermaid = """flowchart TD
  START([Session Start: t=0s]) --> S1["1. OBS Studio: Recording Initialization (11s)"]
  S1 --> S2["2. VS Code: Implementation Plan Review (21s)"]
  S2 --> S3["3. VS Code: Code Inspection & Verification (5s)"]
  S3 --> S4{"Web Verification Needed?"}
  S4 -->|Yes| S5["4. Google Chrome: Table & Doc Lookup (5s)"]
  S5 --> S6["5. Google Chrome: Interactive Verification (12s)"]
  S6 --> S7["6. OBS Studio: Recording Teardown (1s)"]
  S7 --> FINISH([Session Completed: t=55s])

  classDef productive fill:#dcfce7,stroke:#16a34a,stroke-width:2px;
  classDef overhead fill:#fee2e2,stroke:#dc2626,stroke-width:2px;
  classDef lookup fill:#fef3c7,stroke:#d97706,stroke-width:2px;

  class S2,S3 productive;
  class S1,S7 overhead;
  class S5,S6 lookup;
"""

    return WorkflowSynthesisReport(
        session_id=manifest.session_id,
        total_duration_seconds=total_duration,
        executive_summary=(
            f"The analyzed recording session '{manifest.session_id}' spanned {total_duration:.1f} seconds across 3 primary software "
            "environments: OBS Studio, Visual Studio Code, and Google Chrome. The worker's primary operational objective was reviewing "
            "and refining an architectural implementation plan and cross-referencing supporting documentation.\n\n"
            "The workflow demonstrated clear task demarcation: an initial 11-second setup phase in OBS Studio, followed by a concentrated "
            "26-second development and planning phase in VS Code, an interleaved 17-second web documentation verification phase in Google Chrome, "
            "and a 1-second teardown back in OBS Studio.\n\n"
            "Key automation opportunities include replacing manual OBS recording controls with a lightweight background CLI daemon (saving 21.4% "
            "overhead) and deploying a headless browser verification hook to automate web table inspections."
        ),
        macro_steps=steps,
        friction_ledger=frictions,
        automation_specs=automation_specs,
        mermaid_diagram=mermaid,
    )


class Tier2Synthesizer:
    """
    Synthesizes segmented workflow traces into business logic reports,
    interactive Mermaid diagrams, and actionable automation specs using gpt-4o-mini.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def synthesize(
        self,
        manifest: SegmentedWorkflowManifest,
        sop_context: str = "",
        mock: bool = False,
    ) -> WorkflowSynthesisReport:
        """
        Executes workflow synthesis for the given SegmentedWorkflowManifest.
        """
        if mock or not self.api_key or self.api_key.startswith("mock-") or self.api_key == "placeholder_key_for_testing":
            print("[INFO] Tier2Synthesizer: Using mock mode for synthesis.")
            return generate_mock_synthesis(manifest)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)

        # Assemble prompt payload
        segments_summary = []
        for s in manifest.segments:
            segments_summary.append({
                "segment_id": s.segment_id,
                "time_range": f"{s.start_time}s - {s.end_time}s",
                "duration_seconds": s.duration_seconds,
                "primary_application": s.primary_application,
                "window_title": s.primary_window,
                "observed_interaction": s.summary_interaction,
                "cluster_id": s.cluster_id,
                "is_interruption": s.is_interruption,
            })

        clusters_summary = []
        for c in manifest.clusters:
            clusters_summary.append({
                "cluster_id": c.cluster_id,
                "label": c.label,
                "total_duration": c.total_duration_seconds,
                "applications": c.primary_applications,
                "summary": c.representative_summary,
            })

        stitched_summary = []
        for seq in manifest.stitched_sequences:
            stitched_summary.append({
                "sequence_id": seq.sequence_id,
                "macro_task": seq.macro_task_label,
                "primary_segments": seq.primary_segment_ids,
                "interleaved_segments": seq.interleaved_segment_ids,
                "active_seconds": seq.total_active_seconds,
                "interruption_seconds": seq.total_interruption_seconds,
                "notes": seq.reconciliation_notes,
            })

        user_content = {
            "session_id": manifest.session_id,
            "source_video": manifest.source_video,
            "total_segments_count": manifest.total_segments,
            "difference_threshold_used": manifest.difference_threshold,
            "reconstructed_segments": segments_summary,
            "subroutine_clusters": clusters_summary,
            "stitched_macro_sequences": stitched_summary,
            "business_intent_and_sop_context": sop_context or "General software development and technical documentation workflow.",
        }

        print(f"[INFO] Tier2Synthesizer: Requesting gpt-4o-mini synthesis for session '{manifest.session_id}'...")
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_content, indent=2)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        content = response.choices[0].message.content or "{}"
        parsed_json = json.loads(content)

        # Build Pydantic model
        total_duration = sum(s.duration_seconds for s in manifest.segments) if manifest.segments else 56.0
        report = WorkflowSynthesisReport(
            session_id=manifest.session_id,
            total_duration_seconds=total_duration,
            executive_summary=parsed_json.get("executive_summary", "Synthesis completed."),
            macro_steps=[MacroWorkflowStep.model_validate(step) for step in parsed_json.get("macro_steps", [])],
            friction_ledger=[FrictionItem.model_validate(f) for f in parsed_json.get("friction_ledger", [])],
            automation_specs=[AutomationSpec.model_validate(a) for a in parsed_json.get("automation_specs", [])],
            mermaid_diagram=parsed_json.get("mermaid_diagram", "flowchart TD\n  A[Start] --> B[End]"),
        )

        return report


def run_synthesis_pipeline(
    input_manifest_path: Path,
    output_json_path: Path,
    report_markdown_path: Optional[Path] = None,
    mock: bool = False,
    sop_file: Optional[Path] = None,
) -> WorkflowSynthesisReport:
    """
    Loads segmented manifest, executes synthesis, and writes outputs.
    """
    with open(input_manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = SegmentedWorkflowManifest.model_validate(data)

    sop_context = ""
    if sop_file and sop_file.exists():
        with open(sop_file, "r", encoding="utf-8") as f:
            sop_context = f.read()

    synthesizer = Tier2Synthesizer()
    report = synthesizer.synthesize(manifest, sop_context=sop_context, mock=mock)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"[OK] Saved WorkflowSynthesisReport JSON to {output_json_path}")

    if report_markdown_path:
        report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = format_markdown_report(report)
        with open(report_markdown_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[OK] Saved WorkflowSynthesisReport Markdown to {report_markdown_path}")

    return report


def format_markdown_report(report: WorkflowSynthesisReport) -> str:
    """Formats the synthesis report as rich GitHub-flavored markdown."""
    lines = [
        f"# Tier 2 Workflow Synthesis & Automation Blueprint",
        f"",
        f"**Session Identifier**: `{report.session_id}`  ",
        f"**Analyzed Duration**: `{report.total_duration_seconds:.1f} seconds`  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary",
        f"",
        report.executive_summary,
        f"",
        f"---",
        f"",
        f"## 2. Reconstructed Macro-Workflow Flowchart",
        f"",
        f"```mermaid",
        report.mermaid_diagram.strip(),
        f"```",
        f"",
        f"---",
        f"",
        f"## 3. Reconstructed Macro-Workflow Steps",
        f"",
        f"| Step | Title | Application | Time Range | Duration | Inferred Intent & Logic | Inputs | Outputs |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for step in report.macro_steps:
        inputs_str = ", ".join(step.inputs) if step.inputs else "N/A"
        outputs_str = ", ".join(step.outputs) if step.outputs else "N/A"
        lines.append(
            f"| **{step.step_number}** | {step.title} | `{step.application}` | {step.time_range} | {step.duration_seconds}s | {step.intent_and_logic} | {inputs_str} | {outputs_str} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Time-Allocation & Friction Ledger",
        f"",
        f"| Category | Time Observed | % of Session | Severity | Friction Description | Automation Potential |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for f in report.friction_ledger:
        lines.append(
            f"| **{f.category}** | {f.observed_time_seconds}s | {f.percentage_of_session}% | `{f.severity}` | {f.friction_description} | {f.automation_potential} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 5. Production Automation Blueprints",
        f"",
    ])

    for spec in report.automation_specs:
        lines.extend([
            f"### Blueprint: `{spec.blueprint_id}` - {spec.subroutine_name}",
            f"- **Target Friction**: {spec.target_friction}",
            f"- **Automation Type**: `{spec.automation_type}`",
            f"- **Trigger Condition**: {spec.trigger_condition}",
            f"- **Estimated Efficiency Gain**: {spec.estimated_efficiency_gain}",
            f"",
            f"#### Algorithmic Logic Flow:",
        ])
        for s in spec.logic_flow:
            lines.append(f"1. {s}")

        lines.extend([
            f"",
            f"#### Error Handling & Fallbacks:",
        ])
        for e in spec.error_handling:
            lines.append(f"- {e}")

        lines.extend([
            f"",
            f"#### Implementation Code Draft:",
            f"```python",
            spec.implementation_code_draft.strip(),
            f"```",
            f"",
        ])

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tier 2 Workflow Synthesis & Automation Blueprint Generator.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/solix_demo_test/segmented_workflow.json",
        help="Path to segmented_workflow.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/solix_demo_test/workflow_synthesis.json",
        help="Path to output workflow_synthesis.json",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/processed/solix_demo_test/workflow_synthesis_report.md",
        help="Path to output markdown report",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default="",
        help="Optional path to artifact directory markdown copy",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline deterministic mock mode without API calls",
    )
    args = parser.parse_args()

    art_p = Path(args.artifact) if args.artifact else None
    rep = run_synthesis_pipeline(
        input_manifest_path=Path(args.manifest),
        output_json_path=Path(args.output),
        report_markdown_path=Path(args.report),
        mock=args.mock,
    )

    if art_p:
        art_p.parent.mkdir(parents=True, exist_ok=True)
        with open(art_p, "w", encoding="utf-8") as f:
            f.write(format_markdown_report(rep))
        print(f"[OK] Saved copy to artifact directory: {art_p}")

