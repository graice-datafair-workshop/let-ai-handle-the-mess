from states import MultiDatasetAnalysisState
import pandas as pd
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_aws import ChatBedrockConverse
from langgraph.types import interrupt
from dotenv import load_dotenv
from nodes import llm


load_dotenv()


def _parse_json_from_response(content: str):
    """Extract JSON (single object or array) from an LLM response."""
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except Exception:
                continue
    return json.loads(content)


def decide_visualization(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Let the LLM propose multiple visualization options for the user to choose from."""
    print("\n🎨 LLM is proposing visualization options...")

    entity_col = state.get("entity_column", "entity")
    entities = list(state["entity_datasets"].keys())
    analysis_results = state.get("analysis_results", {})
    insights = state.get("insights", [])
    analysis_task = state.get("analysis_task", {})

    entity_stats_cols = list(pd.DataFrame(analysis_results.get("entity_stats", {})).columns) if analysis_results.get("entity_stats") else []
    group_col = analysis_task.get("group_by_column")

    base_prompt = state.get("generated_viz_prompt") or f"""
You are a data visualization expert.
Propose 3 DIFFERENT visualization options for this analysis, ranked by impact.

ANALYSIS PERFORMED: {analysis_task.get('task_name', 'Multi-Entity Analysis')}
ENTITY COLUMN: {entity_col}
ENTITIES: {entities}

KEY INSIGHTS:
{chr(10).join(f'• {insight}' for insight in insights[:5])}

DATA AVAILABLE:
- Entity-level statistics columns: {entity_stats_cols}
- Optional group breakdown by: {group_col or 'N/A'}
- Total records: {analysis_results.get('total_records', 'N/A')}
"""

    full_prompt = base_prompt + f"""

Available visualization types:
- "bar": Simple bar chart (compare one metric across entities)
- "grouped_bar": Grouped bars (compare multiple categories per entity)
- "scatter": Scatter plot (relationship between two metrics)
- "heatmap": Heat map (patterns across two dimensions)
- "boxplot": Box plot (distribution / spread using raw data)
- "pie": Pie chart (composition / proportions)

READABILITY CONSTRAINTS — you MUST follow these:
- There are {len(entities)} entities to display. If that number is > 8, prefer "heatmap" or "bar" (single metric).
- NEVER choose "grouped_bar" if the grouping column has more than 6 unique values — it will be unreadable.
- NEVER choose a chart where the legend would have more than 8 entries.
- For "bar": pick ONE clear y_data metric (e.g. an avg_ column). Do NOT use a column with high cardinality as x_data.
- For "heatmap": it already shows all metrics — just set x_data and y_data to the entity column.
- Prefer simplicity: one insight communicated clearly beats many insights communicated poorly.
- Each of the 3 options MUST use a DIFFERENT viz_type.

Return as a JSON ARRAY of 3 options (best first):
[
  {{
    "viz_type": "bar" | "grouped_bar" | "scatter" | "heatmap" | "boxplot" | "pie",
    "title": "Descriptive title",
    "description": "What this shows and why it matters",
    "x_data": "exact column name for x-axis",
    "y_data": "exact column name for y-axis / values",
    "group_by": "column name for grouping or null",
    "rationale": "Why this visualization is impactful"
  }},
  ...
]
"""

    response = llm.invoke(full_prompt)

    try:
        parsed = _parse_json_from_response(response.content)
        # Handle both array and single-object responses
        if isinstance(parsed, list):
            viz_options = parsed[:3]
        else:
            viz_options = [parsed]
    except Exception:
        first_metric = entity_stats_cols[1] if len(entity_stats_cols) > 1 else "records"
        viz_options = [{
            "viz_type": "bar",
            "title": f"{first_metric.replace('_', ' ').title()} by {entity_col.title()}",
            "description": f"Comparison of {first_metric} across {entity_col}s",
            "x_data": entity_col,
            "y_data": first_metric,
            "group_by": None,
            "rationale": "Basic comparison to understand relative scale"
        }]

    print(f"\n   📊 LLM proposed {len(viz_options)} visualization options:")
    for i, opt in enumerate(viz_options, 1):
        print(f"\n   Option {i}: {opt['viz_type'].upper()}")
        print(f"      Title: {opt['title']}")
        print(f"      Rationale: {opt['rationale']}")

    return {"visualization_options": viz_options}


def choose_visualization(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Interrupt and let the user pick one of the proposed visualization options."""
    viz_options = state.get("visualization_options", [])

    if not viz_options:
        print("⚠️ No visualization options available — skipping choice.")
        return {}

    # Build a user-friendly display for the interrupt
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

    # Parse the user's choice
    try:
        choice = int(str(user_response).strip())
    except ValueError:
        choice = 1  # default to the top-ranked option

    choice = max(1, min(choice, len(viz_options)))
    chosen = viz_options[choice - 1]

    print(f"\n   ✅ You picked Option {choice}: {chosen['viz_type'].upper()} — {chosen['title']}")
    return {"visualization_task": chosen}



def create_visualization(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Create the visualization based on LLM's decision."""
    print("\n🖼️ Creating visualization...")

    viz_task = state["visualization_task"]
    analysis_results = state["analysis_results"]
    entity_col = state.get("entity_column", "entity")

    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

    entity_stats = pd.DataFrame(analysis_results["entity_stats"])
    group_breakdown = pd.DataFrame(analysis_results.get("group_breakdown", {}))
    entities = entity_stats.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 7))
    viz_type = viz_task["viz_type"]

    if viz_type == "bar":
        y_data = viz_task.get("y_data", "")
        numeric_cols = entity_stats.select_dtypes(include="number").columns.tolist()
        if y_data not in numeric_cols:
            y_data = numeric_cols[0] if numeric_cols else None
        if y_data:
            values = entity_stats[y_data].values
            bars = ax.bar(entities, values, color=colors[:len(entities)], edgecolor='white', linewidth=2)
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:,.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 5), textcoords="offset points",
                           ha='center', va='bottom', fontsize=13, fontweight='bold')
            ax.set_ylabel(y_data.replace('_', ' ').title(), fontsize=14)
        ax.set_xlabel(entity_col.title(), fontsize=14)

    elif viz_type == "grouped_bar":
        data = group_breakdown if not group_breakdown.empty else entity_stats.select_dtypes(include="number").iloc[:, :4]
        # Hard cap: keep only top 8 columns by total count to avoid legend explosion
        if len(data.columns) > 8:
            top_cols = data.sum().nlargest(8).index
            data = data[top_cols]
        x = np.arange(len(data.index))
        width = 0.8 / max(len(data.columns), 1)
        for i, col in enumerate(data.columns):
            ax.bar(x + i * width, data[col], width, label=str(col), color=colors[i % len(colors)])
        ax.set_xticks(x + width * (len(data.columns) - 1) / 2)
        ax.set_xticklabels(data.index)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xlabel(entity_col.title(), fontsize=14)

    elif viz_type == "scatter":
        numeric_cols = entity_stats.select_dtypes(include="number").columns.tolist()
        x_col = viz_task.get("x_data") if viz_task.get("x_data") in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        y_col = viz_task.get("y_data") if viz_task.get("y_data") in numeric_cols else (numeric_cols[1] if len(numeric_cols) > 1 else x_col)
        if x_col and y_col:
            x_vals = entity_stats[x_col].values
            y_vals = entity_stats[y_col].values
            sizes = entity_stats["records"].values if "records" in entity_stats.columns else np.ones(len(entities)) * 200
            ax.scatter(x_vals, y_vals, s=sizes / max(sizes.max(), 1) * 1000,
                      c=colors[:len(entities)], alpha=0.7, edgecolors='white', linewidth=2)
            for i, entity in enumerate(entities):
                ax.annotate(entity, (x_vals[i], y_vals[i]),
                           xytext=(10, 5), textcoords='offset points', fontsize=12, fontweight='bold')
            ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=14)
            ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=14)
            ax.text(0.02, 0.98, 'Bubble size = record count',
                   transform=ax.transAxes, fontsize=10, va='top', style='italic', color='gray')

    elif viz_type == "heatmap":
        numeric_cols = entity_stats.select_dtypes(include="number").columns.tolist()
        heatmap_data = entity_stats[numeric_cols].T.astype(float)
        row_min = heatmap_data.min(axis=1).values.reshape(-1, 1)
        row_max = heatmap_data.max(axis=1).values.reshape(-1, 1)
        heatmap_norm = (heatmap_data - row_min) / (row_max - row_min + 1e-9)
        sns.heatmap(heatmap_norm, annot=heatmap_data.values, fmt='.1f',
                   cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalized Value'},
                   linewidths=2, linecolor='white')
        ax.set_yticklabels([c.replace('_', ' ').title() for c in numeric_cols], rotation=0)

    elif viz_type == "boxplot":
        numeric_cols = entity_stats.select_dtypes(include="number").columns.tolist()
        # Hard cap: limit to 12 rows max to avoid dense line noise
        if len(numeric_cols) > 12:
            numeric_cols = numeric_cols[:12]
        heatmap_data = entity_stats[numeric_cols].T.astype(float)
        # Use real raw data — no simulation
        y_col = viz_task.get("y_data")
        data_for_box, labels = [], []
        for entity_name, entity_data in state["entity_datasets"].items():
            df = pd.read_csv(pd.io.common.StringIO(entity_data["cleaned_data"] or entity_data["raw_data"]))
            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                data_for_box.append(df[y_col].dropna().values)
                labels.append(entity_name)
        if data_for_box:
            bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel((y_col or '').replace('_', ' ').title(), fontsize=14)
        ax.set_xlabel(entity_col.title(), fontsize=14)

    elif viz_type == "pie":
        values = entity_stats["records"].values if "records" in entity_stats.columns else entity_stats.iloc[:, 0].values
        wedges, texts, autotexts = ax.pie(
            values, labels=entities, colors=colors[:len(entities)],
            autopct='%1.1f%%', startangle=90, explode=[0.02] * len(entities),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        ax.axis('equal')

    ax.set_title(viz_task["title"], fontsize=18, fontweight='bold', pad=20)
    fig.text(0.5, 0.02, viz_task["description"], ha='center', fontsize=11,
            style='italic', color='#444444', wrap=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    print(f"   ✅ Visualization created: {viz_task['title']}")
    return {"visualization_figure": img_base64}


def export_results(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Display the final analysis results and visualization."""

    entity_col = state.get("entity_column", "entity")

    print("\n" + "═" * 70)
    print("📤 ANALYSIS COMPLETE")
    print("═" * 70)

    print(f"\n📊 ANALYSIS: {state['analysis_task']['task_name']}")

    print(f"\n📋 EXECUTIVE SUMMARY")
    print(f"   {state['summary']}")

    print(f"\n💡 KEY INSIGHTS")
    for i, insight in enumerate(state['insights'], 1):
        print(f"   {i}. {insight}")

    # Display the visualization
    if state.get('visualization_figure'):
        print(f"\n🎨 VISUALIZATION: {state['visualization_task']['title']}")
        print(f"   Why this matters: {state['visualization_task']['rationale']}")

        from IPython.display import Image as IPImage, display as ipy_display
        img_data = base64.b64decode(state['visualization_figure'])
        ipy_display(IPImage(data=img_data))

    print(f"\n📂 {entity_col.upper()}S ANALYZED: {list(state['entity_datasets'].keys())}")
    for entity_name, data in state['entity_datasets'].items():
        print(f"   • {entity_name}: {data['row_count']:,} records")

    print(f"\n🧹 CLEANING ACTIONS ({len(state['cleaning_actions'])} total)")
    for action in state['cleaning_actions'][:8]:
        print(f"   {action}")
    if len(state['cleaning_actions']) > 8:
        print(f"   ... and {len(state['cleaning_actions']) - 8} more")

    print(f"\n👤 YOUR PREFERENCES (applied to ALL {entity_col}s)")
    for pref in state['user_preferences']:
        print(f"   • {pref['column']}: {pref['choice']}")

    print("\n" + "═" * 70)
    print("✅ Done! Preferences applied consistently. LLM chose analysis AND visualization.")
    print("═" * 70)

    return {}