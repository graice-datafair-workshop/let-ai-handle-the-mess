from typing import Annotated, Literal, TypedDict
import operator


# 🔍 Issue and Preference structures
class DetectedIssue(TypedDict):
    """A single detected data quality issue"""
    issue_type: Literal["missing_values", "currency_format", "boolean_encoding", "outliers", "date_format"]
    column: str
    description: str
    examples: list[str]
    options: list[str]
    severity: Literal["high", "medium", "low"]


class UserPreference(TypedDict):
    """User's choice for handling an issue"""
    issue_type: str
    column: str
    choice: str


class EntityData(TypedDict):
    """Data for a single entity (city, country, store, etc.)"""
    entity_name: str
    raw_data: str        # CSV string
    cleaned_data: str | None
    row_count: int


class AnalysisTask(TypedDict):
    """The task the LLM decided to perform"""
    task_name: str
    task_description: str
    required_columns: list[str]
    analysis_type: Literal["comparison", "ranking", "trends", "segmentation", "distribution"]
    group_by_column: str | None   # LLM suggests what to group by, if anything


class VisualizationTask(TypedDict):
    """The visualization the LLM decided to create"""
    viz_type: Literal["bar", "grouped_bar", "scatter", "heatmap", "boxplot", "pie"]
    title: str
    description: str
    x_data: str       # Column or aggregation for x-axis
    y_data: str       # Column or aggregation for y-axis
    group_by: str | None
    rationale: str


# 📊 The full agent state
class MultiDatasetAnalysisState(TypedDict):
    # --- Config (set once at startup) ---
    entity_column: str            # e.g. "city", "country", "store" — the partitioning dimension

    # --- Input ---
    entities_to_analyze: list[str]

    # --- Data Storage ---
    entity_datasets: dict[str, EntityData]   # entity_name -> data

    # --- Issue Detection (from first entity, applies to all) ---
    detected_issues: list[DetectedIssue] | None

    # --- User Preferences — SHARED ACROSS ALL ENTITIES ---
    user_preferences: list[UserPreference]
    preferences_collected: bool

    # --- Cleaning Results ---
    cleaning_actions: Annotated[list[str], operator.add]

    # --- Meta-prompting (LLM writes prompts for the dataset) ---
    generated_analysis_prompt: str | None
    generated_viz_prompt: str | None

    # --- LLM-Proposed Analysis Options ---
    analysis_options: list[AnalysisTask] | None  # Multiple proposals from LLM
    analysis_task: AnalysisTask | None            # The one the user picked

    # --- LLM-Proposed Visualization Options ---
    visualization_options: list[VisualizationTask] | None  # Multiple proposals from LLM
    visualization_task: VisualizationTask | None            # The one the user picked
    visualization_figure: str | None   # Base64 encoded image

    # --- Final Results ---
    analysis_results: dict | None
    insights: list[str] | None
    summary: str | None
