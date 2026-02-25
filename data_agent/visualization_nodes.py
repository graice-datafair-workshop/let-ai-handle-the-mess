import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image as IPImage, display as ipy_display
from langgraph.types import interrupt

from .llm_setup import llm
from .states import CombinedAgentState
from .helper_functions import _parse_json_from_response, unique_file_label


# =============================================================================
# NODE: DECIDE VISUALIZATION
# =============================================================================

def decide_visualization(state: CombinedAgentState) -> dict:
    """Let the LLM propose multiple visualization options for the user to choose from."""
    print("\nLLM is proposing visualization options...")

    file_col = "source_file"
    file_states = state.get('file_states', {})
    all_paths = list(file_states.keys())
    file_names = [unique_file_label(fp, all_paths) for fp in all_paths]
    analysis_results = state.get("analysis_results", {})
    insights = state.get("insights", [])
    analysis_task = state.get("analysis_task", {})

    stats_df = pd.DataFrame(analysis_results.get("entity_stats", {}))
    file_stats_cols = list(stats_df.columns) if not stats_df.empty else []
    group_col = analysis_task.get("group_by_column")

    insights_str = '\n'.join(f'• {insight}' for insight in insights[:5])

    base_prompt = (
        f"You are a data visualization expert.\n"
        f"Propose 3 DIFFERENT visualization options for this analysis, ranked by impact.\n\n"
        f"ANALYSIS PERFORMED: {analysis_task.get('task_name', 'Multi-File Analysis')}\n"
        f"FILE COLUMN: {file_col}\n"
        f"FILES: {file_names}\n\n"
        f"KEY INSIGHTS:\n{insights_str}\n\n"
        f"DATA AVAILABLE:\n"
        f"- File-level statistics columns: {file_stats_cols}\n"
        f"- Optional group breakdown by: {group_col or 'N/A'}\n"
        f"- Total records: {analysis_results.get('total_records', 'N/A')}\n"
    )

    suffix_prompt = (
        f"\nAvailable visualization types:\n"
        f"- \"bar\": Simple bar chart (compare one metric across files)\n"
        f"- \"grouped_bar\": Grouped bars (compare multiple categories per file)\n"
        f"- \"scatter\": Scatter plot (relationship between two metrics)\n"
        f"- \"heatmap\": Heat map (patterns across two dimensions)\n"
        f"- \"boxplot\": Box plot (distribution / spread using raw data)\n"
        f"- \"pie\": Pie chart (composition / proportions)\n\n"
        f"READABILITY CONSTRAINTS — you MUST follow these:\n"
        f"- There are {len(file_names)} files to display. If that number is > 8, prefer \"heatmap\" or \"bar\".\n"
        f"- NEVER choose \"grouped_bar\" if the grouping column has more than 6 unique values.\n"
        f"- NEVER choose a chart where the legend would have more than 8 entries.\n"
        f"- For \"bar\": pick ONE clear y_data metric. Do NOT use a column with high cardinality as x_data.\n"
        f"- Prefer simplicity: one insight communicated clearly beats many insights communicated poorly.\n"
        f"- Each of the 3 options MUST use a DIFFERENT viz_type.\n\n"
        "Return as a JSON ARRAY of 3 options (best first):\n"
        "[\n"
        "  {\n"
        "    \"viz_type\": \"bar\" | \"grouped_bar\" | \"scatter\" | \"heatmap\" | \"boxplot\" | \"pie\",\n"
        "    \"title\": \"Descriptive title\",\n"
        "    \"description\": \"What this shows and why it matters\",\n"
        "    \"x_data\": \"exact column name for x-axis\",\n"
        "    \"y_data\": \"exact column name for y-axis / values\",\n"
        "    \"group_by\": \"column name for grouping or null\",\n"
        "    \"rationale\": \"Why this visualization is impactful\"\n"
        "  },\n"
        "  ...\n"
        "]"
    )
    full_prompt = base_prompt + suffix_prompt

    response = llm.invoke(full_prompt)

    try:
        parsed = _parse_json_from_response(response.content)
        if isinstance(parsed, list):
            viz_options = parsed[:3]
        else:
            viz_options = [parsed]
    except Exception:
        first_metric = file_stats_cols[1] if len(file_stats_cols) > 1 else "records"
        viz_options = [{
            "viz_type": "bar",
            "title": f"{first_metric.replace('_', ' ').title()} by File",
            "description": f"Comparison of {first_metric} across files",
            "x_data": file_col,
            "y_data": first_metric,
            "group_by": None,
            "rationale": "Basic comparison to understand relative scale"
        }]

    print(f"\n   LLM proposed {len(viz_options)} visualization options:")
    for i, opt in enumerate(viz_options, 1):
        print(f"\n   Option {i}: {opt['viz_type'].upper()}")
        print(f"      Title: {opt['title']}")
        print(f"      Rationale: {opt['rationale']}")

    return {"visualization_options": viz_options}


# =============================================================================
# NODE: CHOOSE VISUALIZATION
# =============================================================================

def choose_visualization(state: CombinedAgentState) -> dict:
    """Interrupt and let the user pick one of the proposed visualization options."""
    viz_options = state.get("visualization_options", [])

    if not viz_options:
        print("No visualization options available — skipping choice.")
        return {}

    options_display = []
    for i, opt in enumerate(viz_options, 1):
        options_display.append({
            "option": i,
            "type": opt["viz_type"],
            "title": opt["title"],
            "description": opt["description"],
            "rationale": opt["rationale"],
        })

    user_response = interrupt({
        "message": "Which visualization would you like? Pick a number (1, 2, or 3).",
        "options": options_display,
    })

    try:
        choice = int(str(user_response).strip())
    except ValueError:
        choice = 1

    choice = max(1, min(choice, len(viz_options)))
    chosen = viz_options[choice - 1]

    print(f"\n   You picked Option {choice}: {chosen['viz_type'].upper()} — {chosen['title']}")
    return {"visualization_task": chosen}


# =============================================================================
# NODE: CREATE VISUALIZATION
# =============================================================================

def create_visualization(state: CombinedAgentState) -> dict:
    """Create the visualization based on LLM's decision."""
    print("\nCreating visualization...")

    viz_task = state["visualization_task"]
    analysis_results = state["analysis_results"]
    file_col = "source_file"

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

    file_stats = pd.DataFrame(analysis_results["entity_stats"])
    group_breakdown = pd.DataFrame(analysis_results.get("group_breakdown", {}))
    files = file_stats.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    viz_type = viz_task["viz_type"]

    if viz_type == "bar":
        y_data = viz_task.get("y_data", "")
        numeric_cols = file_stats.select_dtypes(include="number").columns.tolist()
        if y_data not in numeric_cols:
            y_data = numeric_cols[0] if numeric_cols else None
        if y_data:
            values = file_stats[y_data].values
            bars = ax.bar(
                files, values, color=colors[:len(files)],
                edgecolor='white', linewidth=2
            )
            for bar, val in zip(bars, values):
                ax.annotate(
                    f'{val:,.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=13, fontweight='bold'
                )
            ax.set_ylabel(y_data.replace('_', ' ').title(), fontsize=14)
        ax.set_xlabel("File", fontsize=14)

    elif viz_type == "grouped_bar":
        data = (
            group_breakdown if not group_breakdown.empty
            else file_stats.select_dtypes(include="number").iloc[:, :4]
        )
        if len(data.columns) > 8:
            top_cols = data.sum().nlargest(8).index
            data = data[top_cols]
        x = np.arange(len(data.index))
        width = 0.8 / max(len(data.columns), 1)
        for i, col in enumerate(data.columns):
            ax.bar(
                x + i * width, data[col], width,
                label=str(col), color=colors[i % len(colors)]
            )
        ax.set_xticks(x + width * (len(data.columns) - 1) / 2)
        ax.set_xticklabels(data.index)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlabel("File", fontsize=14)

    elif viz_type == "scatter":
        numeric_cols = file_stats.select_dtypes(include="number").columns.tolist()
        x_col = (
            viz_task.get("x_data") if viz_task.get("x_data") in numeric_cols
            else (numeric_cols[0] if numeric_cols else None)
        )
        y_col = (
            viz_task.get("y_data") if viz_task.get("y_data") in numeric_cols
            else (numeric_cols[1] if len(numeric_cols) > 1 else x_col)
        )
        if x_col and y_col:
            x_vals = file_stats[x_col].values
            y_vals = file_stats[y_col].values
            sizes = (
                file_stats["records"].values if "records" in file_stats.columns
                else np.ones(len(files)) * 200
            )
            ax.scatter(
                x_vals, y_vals, s=sizes / max(sizes.max(), 1) * 1000,
                c=colors[:len(files)], alpha=0.7, edgecolors='white', linewidth=2
            )
            for i, file_name in enumerate(files):
                ax.annotate(
                    file_name, (x_vals[i], y_vals[i]),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold'
                )
            ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=14)
            ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=14)
            ax.text(
                0.02, 0.98, 'Bubble size = record count',
                transform=ax.transAxes, fontsize=10, va='top',
                style='italic', color='gray'
            )

    elif viz_type == "heatmap":
        numeric_cols = file_stats.select_dtypes(include="number").columns.tolist()
        heatmap_data = file_stats[numeric_cols].T.astype(float)
        row_min = heatmap_data.min(axis=1).values.reshape(-1, 1)
        row_max = heatmap_data.max(axis=1).values.reshape(-1, 1)
        heatmap_norm = (heatmap_data - row_min) / (row_max - row_min + 1e-9)
        sns.heatmap(
            heatmap_norm, annot=heatmap_data.values, fmt='.1f',
            cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalized Value'},
            linewidths=2, linecolor='white'
        )
        ax.set_yticklabels(
            [c.replace('_', ' ').title() for c in numeric_cols], rotation=0
        )

    elif viz_type == "boxplot":
        y_col = viz_task.get("y_data")
        data_for_box, labels = [], []
        file_states = state.get('file_states', {})
        all_paths = list(file_states.keys())
        for file_path, file_state in file_states.items():
            df = file_state['working_df']
            file_name = unique_file_label(file_path, all_paths)
            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                data_for_box.append(df[y_col].dropna().values)
                labels.append(file_name)
        if data_for_box:
            bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel((y_col or '').replace('_', ' ').title(), fontsize=14)
        ax.set_xlabel("File", fontsize=14)

    elif viz_type == "pie":
        values = (
            file_stats["records"].values if "records" in file_stats.columns
            else file_stats.iloc[:, 0].values
        )
        wedges, texts, autotexts = ax.pie(
            values, labels=files, colors=colors[:len(files)],
            autopct='%1.1f%%', startangle=90, explode=[0.02] * len(files),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        ax.axis('equal')

    ax.set_title(viz_task["title"], fontsize=18, fontweight='bold', pad=20)
    fig.text(
        0.5, 0.02, viz_task["description"], ha='center', fontsize=11,
        style='italic', color='#444444', wrap=True
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    buffer = BytesIO()
    plt.savefig(
        buffer, format='png', dpi=150, bbox_inches='tight',
        facecolor='white', edgecolor='none'
    )
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    print(f"   Visualization created: {viz_task['title']}")
    return {"visualization_figure": img_base64}


# =============================================================================
# NODE: EXPORT RESULTS
# =============================================================================

def export_results(state: CombinedAgentState) -> dict:
    """Display the final analysis results and visualization."""

    print("\n" + "═" * 70)
    print("ANALYSIS COMPLETE")
    print("═" * 70)

    print(f"\nANALYSIS: {state['analysis_task']['task_name']}")

    print("\nEXECUTIVE SUMMARY")
    print(f"   {state['summary']}")

    print("\nKEY INSIGHTS")
    for i, insight in enumerate(state['insights'], 1):
        print(f"   {i}. {insight}")

    # Display the visualization
    if state.get('visualization_figure'):
        print(f"\nVISUALIZATION: {state['visualization_task']['title']}")
        print(f"Why this matters: {state['visualization_task']['rationale']}")

        img_data = base64.b64decode(state['visualization_figure'])
        ipy_display(IPImage(data=img_data))

    # Display file info from file_states
    file_states = state.get('file_states', {})
    all_paths = list(file_states.keys())
    file_names = [unique_file_label(fp, all_paths) for fp in all_paths]
    print(f"\nFILES ANALYZED: {file_names}")
    for file_path, file_state in file_states.items():
        file_name = unique_file_label(file_path, all_paths)
        row_count = len(file_state['working_df'])
        print(f"   • {file_name}: {row_count:,} records")

    # Display processing log
    processing_log = state.get('processing_log', [])
    if processing_log:
        print(f"\nPROCESSING LOG ({len(processing_log)} entries)")
        for log_entry in processing_log[:8]:
            print(f"   {log_entry}")
        if len(processing_log) > 8:
            print(f"   ... and {len(processing_log) - 8} more")

    print("\n" + "═" * 70)
    print("Done! LLM chose analysis AND visualization for cleaned data.")
    print("═" * 70)

    return {}
