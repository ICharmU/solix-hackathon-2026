"""
Data schemas for multimodal temporal alignment, frame extraction, transcription,
Tier 1 visual descriptors, and text embeddings.
"""

from typing import List, Optional, Dict, Any
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


# --- Phase 3: Boundaries, Clustering & Stitching Schemas ---

class TaskSegment(BaseModel):
    """Contiguous timeline segment bounded by state derivative breakpoints or application shifts."""
    segment_id: str = Field(..., description="Unique segment identifier e.g. seg_01")
    start_time: int = Field(..., description="Start timestamp in seconds (inclusive)")
    end_time: int = Field(..., description="End timestamp in seconds (inclusive)")
    duration_seconds: int = Field(..., description="Total duration of this segment in seconds")
    primary_application: str = Field(default="Unknown", description="Dominant active application")
    primary_window: str = Field(default="", description="Dominant window title")
    summary_interaction: str = Field(default="", description="Consolidated description of user interaction")
    centroid_embedding: Optional[List[float]] = Field(default=None, description="Average unit-normalized embedding vector")
    cluster_id: Optional[str] = Field(default=None, description="Assigned cluster identifier")
    is_interruption: bool = Field(default=False, description="Whether this segment represents an interleaved interruption")
    interrupted_segment_id: Optional[str] = Field(default=None, description="Segment ID that this segment interrupted, if applicable")
    slots_count: int = Field(default=0, description="Number of 1-second frames in this segment")
    slot_timestamps: List[int] = Field(default_factory=list, description="List of timestamps contained in this segment")

    # Neural CUDA Context Tracking Signals
    neural_transition_type: Optional[str] = Field(default=None, description="Predicted transition type (NEW_MACRO, SUBROUTINE_SHIFT, RESUME)")
    neural_confidence: Optional[float] = Field(default=None, description="Confidence score from TaskDifferentiationNet")
    resumed_segment_id: Optional[str] = Field(default=None, description="Pointer to interrupted segment being resumed")
    continuation_probability: Optional[float] = Field(default=None, description="TaskContinuationNet confidence score")


class SubroutineCluster(BaseModel):
    """Cluster of semantically similar task segments across the session."""
    cluster_id: str = Field(..., description="Unique cluster identifier e.g. cluster_01")
    label: str = Field(..., description="Human-readable cluster label e.g. Code Development")
    segment_ids: List[str] = Field(default_factory=list, description="IDs of segments assigned to this cluster")
    total_duration_seconds: int = Field(default=0, description="Sum of segment durations in this cluster")
    primary_applications: List[str] = Field(default_factory=list, description="Applications present in this cluster")
    representative_summary: str = Field(default="", description="High-level summary of the subroutine behavior")


class StitchedSequence(BaseModel):
    """Reconciled macro-routine with interleaved interruptions stitched into a cohesive workflow."""
    sequence_id: str = Field(..., description="Unique sequence identifier e.g. seq_01")
    macro_task_label: str = Field(..., description="Label of the primary continuous task")
    primary_segment_ids: List[str] = Field(default_factory=list, description="Segments belonging to the core task (e.g. A1, A2)")
    interleaved_segment_ids: List[str] = Field(default_factory=list, description="Segments identified as interruptions (e.g. B)")
    total_active_seconds: int = Field(default=0, description="Duration spent on core task")
    total_interruption_seconds: int = Field(default=0, description="Duration spent on interruptions")
    reconciliation_notes: str = Field(default="", description="Explanation of why segments were stitched")


class SegmentedWorkflowManifest(BaseModel):
    """Complete Phase 3 manifest capturing segments, subroutine clusters, and stitched workflows."""
    session_id: str
    source_video: str
    created_at: str
    difference_threshold: float = Field(default=0.5, description="Breakpoint heuristic threshold used")
    total_segments: int
    total_clusters: int
    total_stitched_sequences: int
    document_context: Optional[Dict[str, Any]] = Field(default=None, description="Upfront project PRD, active tickets, schemas, and constraints")
    segments: List[TaskSegment] = Field(default_factory=list)
    clusters: List[SubroutineCluster] = Field(default_factory=list)
    stitched_sequences: List[StitchedSequence] = Field(default_factory=list)


# --- Phase 4: Tier 2 Workflow Synthesis & Automation Specs ---

class MacroWorkflowStep(BaseModel):
    """Reconstructed macro step in the business process."""
    step_number: int = Field(..., description="Chronological step index")
    title: str = Field(..., description="High-level descriptive step name")
    application: str = Field(..., description="Primary software application used")
    time_range: str = Field(..., description="Interval string e.g. 11s - 31s")
    duration_seconds: int = Field(..., description="Duration in seconds")
    intent_and_logic: str = Field(..., description="Inferred business intent, decisions made, or operations performed")
    inputs: List[str] = Field(default_factory=list, description="Documents, forms, tabs, or data sources read")
    outputs: List[str] = Field(default_factory=list, description="Artifacts, edits, entries, or decisions produced")


class FrictionItem(BaseModel):
    """Quantified workflow friction or bottleneck entry."""
    category: str = Field(..., description="Friction type e.g. Manual Setup, Context Switching, Repetitive Verification")
    observed_time_seconds: int = Field(..., description="Time spent on this friction activity")
    percentage_of_session: float = Field(..., description="Percentage of total recording time")
    severity: str = Field(default="Medium", description="Low, Medium, or High bottleneck impact")
    friction_description: str = Field(..., description="Detailed description of the inefficiency observed")
    automation_potential: str = Field(..., description="Feasibility and strategy for automating or eliminating this friction")


class AutomationSpec(BaseModel):
    """Actionable specification for automating repetitive subroutines."""
    blueprint_id: str = Field(..., description="Identifier e.g. AUTO-01")
    subroutine_name: str = Field(..., description="Name of the routine to automate")
    target_friction: str = Field(..., description="Friction item addressed")
    automation_type: str = Field(..., description="e.g. Python Script, Playwright RPA, Browser Automation, API Hook")
    trigger_condition: str = Field(..., description="When this automation should execute")
    logic_flow: List[str] = Field(default_factory=list, description="Step-by-step algorithmic logic")
    error_handling: List[str] = Field(default_factory=list, description="Edge cases, timeouts, fallback strategies")
    estimated_efficiency_gain: str = Field(..., description="Projected time savings or friction reduction")
    implementation_code_draft: str = Field(..., description="Production-ready starter code or configuration")


class WorkflowSynthesisReport(BaseModel):
    """Complete Tier 2 synthesis deliverable combining business logic, ledger, and automation specs."""
    session_id: str
    total_duration_seconds: float
    executive_summary: str = Field(..., description="Executive overview of the session, goals, and outcomes")
    macro_steps: List[MacroWorkflowStep] = Field(default_factory=list)
    friction_ledger: List[FrictionItem] = Field(default_factory=list)
    automation_specs: List[AutomationSpec] = Field(default_factory=list)
    mermaid_diagram: str = Field(..., description="Renderable Mermaid flowchart string")


