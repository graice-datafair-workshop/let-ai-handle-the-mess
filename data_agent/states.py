import operator
from typing import Annotated, TypedDict

import pandas as pd


class ColumnInfo(TypedDict):
    """Information about a single column's type inference."""
    column_name: str
    inferred_type: str
    reasoning: str
    sample_values: list[str]
    confidence: float
    user_approved: bool | None  # None = not yet reviewed
    user_correction: str | None  # User's requested type if they disagree
    transformation_applied: bool


class FileState(TypedDict):
    """State for a single file being processed."""
    file_path: str
    original_df: pd.DataFrame | None
    working_df: pd.DataFrame | None
    column_names: list[str]
    current_column_index: int
    column_info: dict[str, ColumnInfo]
    status: str


class AnalysisTask(TypedDict):
    """A proposed analysis task."""
    task_name: str
    task_description: str
    group_by_column: str | None


class VisualizationTask(TypedDict):
    """A proposed visualization task."""
    viz_type: str
    title: str
    description: str
    x_data: str
    y_data: str
    group_by: str | None
    rationale: str


class CombinedAgentState(TypedDict):
    """Complete state for the combined data cleaning and analysis agent."""
    # ===== Multi-file Processing =====
    files_to_process: list[str]
    current_file_index: int
    file_states: dict[str, FileState]

    # ===== Apply-to-all Decisions =====
    apply_decisions_to_all_files: bool
    decisions_preloaded: bool  # True if decisions were loaded from file (apply to first file too)
    stored_column_decisions: dict[str, dict]
    stored_duplicate_decision: bool | None
    stored_nil_decisions: dict[str, str]
    stored_columns_to_drop: list[str]

    # ===== Processing Log =====
    processing_log: Annotated[list[str], operator.add]

    # ===== Status Tracking =====
    status: str
    _nil_columns_processed: list[str]

    # ===== Analysis Phase =====
    generated_analysis_prompt: str | None
    analysis_options: list[AnalysisTask] | None
    analysis_task: AnalysisTask | None
    analysis_results: dict | None
    insights: list[str] | None
    summary: str | None

    # ===== Visualization Phase =====
    visualization_options: list[VisualizationTask] | None
    visualization_task: VisualizationTask | None
    visualization_figure: str | None
