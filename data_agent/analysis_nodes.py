import json
from pathlib import Path

import numpy as np
import pandas as pd
from langgraph.types import interrupt

from .llm_setup import llm
from .states import CombinedAgentState
from .helper_functions import _parse_json_from_response, _make_json_serializable, unique_file_label


# =============================================================================
# NODE: GENERATE ANALYSIS PROMPT
# =============================================================================

def generate_analysis_prompt(state: CombinedAgentState) -> dict:
    """Use a fast/cheap LLM to write a domain-specific prompt for the analysis step."""

    sample_rows = []
    column_profiles = {}
    file_names = []

    file_states = state.get('file_states', {})
    all_paths = list(file_states.keys())
    for file_path, file_state in file_states.items():
        df = file_state['working_df']
        file_names.append(unique_file_label(file_path, all_paths))
        # Convert to JSON-serializable format (handles Timestamps, etc.)
        sample_rows = _make_json_serializable(df.head(3).to_dict(orient="records"))

        for col in df.columns:
            # Convert sample values to JSON-serializable format
            raw_samples = df[col].dropna().head(5).tolist()
            column_profiles[col] = {
                "dtype": str(df[col].dtype),
                "nulls_pct": round(df[col].isnull().mean() * 100, 1),
                "sample_values": _make_json_serializable(raw_samples),
                "unique_count": int(df[col].nunique()),
            }
        break  # profile from first file; similar schema expected across all

    meta_prompt = (
        f"You are a prompt engineer. Your task is to write a precise LLM prompt "
        f"that will instruct another LLM to choose the best analysis for this dataset.\n\n"
        f"DATASET SCHEMA:\n{json.dumps(column_profiles, indent=2)}\n\n"
        f"SAMPLE ROWS:\n{json.dumps(sample_rows, indent=2)}\n\n"
        f"FILES AVAILABLE: {file_names}\n\n"
        f"Write a prompt (plain text, no JSON) that:\n"
        f"1. Describes what this dataset appears to contain (infer the domain)\n"
        f"2. Lists the most analytically interesting columns and why\n"
        f"3. Asks the LLM to pick one analysis type and return a JSON task spec "
        f"with fields: task_name, task_description, required_columns, analysis_type, group_by_column\n"
        f"Keep the prompt under 500 words."
    )

    response = llm.invoke(meta_prompt)
    generated_prompt = response.content

    print(f"\nGenerated analysis prompt ({len(generated_prompt)} chars)")

    return {"generated_analysis_prompt": generated_prompt}


# =============================================================================
# NODE: DECIDE ANALYSIS TASK
# =============================================================================

def decide_analysis_task(state: CombinedAgentState) -> dict:
    """Let the LLM propose multiple analysis options for the user to choose from."""
    print("\nLLM is proposing analysis options...")

    files_info = []
    all_columns = set()

    file_states = state.get('file_states', {})
    all_paths = list(file_states.keys())
    for file_path, file_state in file_states.items():
        df = file_state['working_df']
        file_name = unique_file_label(file_path, all_paths)
        summary = {"file": file_name, "rows": len(df)}
        for col in df.select_dtypes(include="number").columns:
            summary[f"avg_{col}"] = round(df[col].mean(), 2)
        files_info.append(summary)
        all_columns.update(df.columns.tolist())

    file_names = [f["file"] for f in files_info]

    default_prompt = (
        f"You are a data analyst. You have datasets from these files: {file_names}\n"
        f"Available columns: {list(all_columns)}\n"
        f"Data summary:\n{json.dumps(files_info, indent=2)}\n\n"
        f"Propose 3 DIFFERENT analysis tasks that would provide the most valuable insights."
    )
    base_prompt = state.get("generated_analysis_prompt") or default_prompt

    suffix_prompt = (
        "\n\nPropose 3 DIFFERENT analysis tasks, ranked by value. Each must focus on a different angle.\n\n"
        "Return as a JSON ARRAY of 3 options (best first, inside a ```json block):\n"
        "[\n"
        "  {\n"
        "    \"task_name\": \"Short name for the analysis\",\n"
        "    \"task_description\": \"What exactly you will analyze and why it's valuable\",\n"
        "    \"required_columns\": [\"col1\", \"col2\"],\n"
        "    \"analysis_type\": \"comparison\" | \"ranking\" | \"trends\" | \"segmentation\" | \"distribution\",\n"
        "    \"group_by_column\": null\n"
        "  },\n"
        "  ...\n"
        "]"
    )
    full_prompt = base_prompt + suffix_prompt

    response = llm.invoke(full_prompt)

    try:
        parsed = _parse_json_from_response(response.content)
        if isinstance(parsed, list):
            analysis_options = parsed[:3]
        else:
            analysis_options = [parsed]
    except Exception:
        analysis_options = [{
            "task_name": "Cross-File Comparison",
            "task_description": "Compare key metrics across all files",
            "required_columns": list(all_columns)[:5],
            "analysis_type": "comparison",
            "group_by_column": None
        }]

    print(f"\n   LLM proposed {len(analysis_options)} analysis options:")
    for i, opt in enumerate(analysis_options, 1):
        print(f"\n   Option {i}: {opt['task_name']}")
        print(f"      {opt['task_description']}")

    return {"analysis_options": analysis_options}


# =============================================================================
# NODE: CHOOSE ANALYSIS TASK
# =============================================================================

def choose_analysis_task(state: CombinedAgentState) -> dict:
    """Interrupt and let the user pick one of the proposed analysis options."""
    analysis_options = state.get("analysis_options", [])

    if not analysis_options:
        print("No analysis options available — skipping choice.")
        return {}

    options_display = []
    for i, opt in enumerate(analysis_options, 1):
        options_display.append({
            "option": i,
            "task_name": opt["task_name"],
            "task_description": opt["task_description"],
            "analysis_type": opt.get("analysis_type", "comparison"),
        })

    user_response = interrupt({
        "message": "Which analysis would you like? Pick a number (1, 2, or 3).",
        "options": options_display,
    })

    try:
        choice = int(str(user_response).strip())
    except ValueError:
        choice = 1

    choice = max(1, min(choice, len(analysis_options)))
    chosen = analysis_options[choice - 1]

    print(f"\n   You picked Option {choice}: {chosen['task_name']}")
    print(f"   {chosen['task_description']}")
    return {"analysis_task": chosen}


# =============================================================================
# NODE: EXECUTE ANALYSIS
# =============================================================================

def execute_analysis(state: CombinedAgentState) -> dict:
    """Execute the LLM-decided analysis across all files."""
    print(f"\nExecuting: {state['analysis_task']['task_name']}...")

    file_col = "source_file"
    task = state['analysis_task']

    # Combine all file DataFrames from file_states
    all_dfs = []
    file_states = state.get('file_states', {})
    all_paths = list(file_states.keys())
    for file_path, file_state in file_states.items():
        df = file_state['working_df'].copy()
        df[file_col] = unique_file_label(file_path, all_paths)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Keep only task-relevant columns (+ file label) to stay within LLM input limits
    required_cols = [c for c in task.get("required_columns", []) if c in combined_df.columns]
    group_col = task.get("group_by_column")
    cols_to_keep = list(dict.fromkeys(
        [file_col]
        + required_cols
        + ([group_col] if group_col and group_col in combined_df.columns else [])
    ))
    # Fallback: if none of the required columns matched, use all columns but cap at 8
    if len(cols_to_keep) <= 1:
        cols_to_keep = [file_col] + [c for c in combined_df.columns if c != file_col][:8]

    slim_df = combined_df[cols_to_keep]

    # Build a compact summary: 10-row sample per file + basic stats
    sample_parts = []
    stats_parts = []
    for label, grp in slim_df.groupby(file_col):
        sample_parts.append(f"--- {label} ({len(grp)} rows total) ---")
        sample_parts.append(grp.head(10).to_string(index=False))

        num_cols = grp.select_dtypes(include="number").columns.tolist()
        if num_cols:
            desc = grp[num_cols].describe().loc[["mean", "50%", "min", "max"]].round(2)
            stats_parts.append(f"--- {label} ---")
            stats_parts.append(desc.to_string())

    samples_str = "\n".join(sample_parts)
    stats_str = "\n".join(stats_parts) if stats_parts else "No numeric columns."

    group_breakdown_str = "N/A"
    group_breakdown_dict = {}
    if group_col and group_col in combined_df.columns:
        group_breakdown = combined_df.groupby(
            [file_col, group_col]
        ).size().unstack(fill_value=0)
        # Limit to top 15 categories to keep prompt small
        if len(group_breakdown.columns) > 15:
            top_cats = combined_df[group_col].value_counts().head(15).index
            group_breakdown = group_breakdown[top_cats]
        group_breakdown_str = group_breakdown.to_string()
        group_breakdown_dict = group_breakdown.to_dict()

    analysis_prompt = (
        f"You are executing the analysis: \"{task['task_name']}\"\n"
        f"Task: {task['task_description']}\n\n"
        f"DATA (10-row sample per file):\n"
        f"{samples_str}\n\n"
        f"SUMMARY STATISTICS:\n"
        f"{stats_str}\n\n"
        f"BREAKDOWN BY '{group_col or 'N/A'}':\n"
        f"{group_breakdown_str}\n\n"
        f"Based on this data, provide:\n"
        f"1. 5-7 specific, actionable insights that answer the analysis goal\n"
        f"2. A 2-3 sentence executive summary\n\n"
        f"Format as JSON:\n"
        "{{\n"
        "    \"insights\": [\"insight 1\", \"insight 2\", ...],\n"
        "    \"summary\": \"Executive summary here\"\n"
        "}}"
    )

    response = llm.invoke(analysis_prompt)

    try:
        content = response.content
        result = _parse_json_from_response(content)
        insights = result.get("insights", [])
        summary = result.get("summary", "Analysis complete.")
    except Exception:
        insights = [response.content]
        summary = "Analysis complete."

    print(f"   Generated {len(insights)} insights")
    # Build lightweight stats dict for the return value
    numeric_cols = slim_df.select_dtypes(include="number").columns.tolist()
    entity_stats = (
        slim_df.groupby(file_col)[numeric_cols].mean().round(2).to_dict()
        if numeric_cols else {}
    )
    return {
        "analysis_results": {
            "entity_stats": entity_stats,
            "group_breakdown": group_breakdown_dict,
            "total_records": len(combined_df)
        },
        "insights": insights,
        "summary": summary
    }
