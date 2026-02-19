from states import MultiDatasetAnalysisState
from dotenv import load_dotenv
import pandas as pd
import json
from langchain_aws import ChatBedrockConverse
from dataset_config import DATASET_CONFIG
from langgraph.types import interrupt
import os
import boto3
from botocore.config import Config
from botocore import UNSIGNED
from langchain_aws import ChatBedrockConverse

load_dotenv()

token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    config=Config(signature_version=UNSIGNED),
)

def add_bearer_token(request, **kwargs):
    request.headers["Authorization"] = f"Bearer {token}"

bedrock_client.meta.events.register("before-send.bedrock-runtime.*", add_bearer_token)

llm = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    client=bedrock_client,
)

# Quick test
response = llm.invoke("Say 'Ready to analyze data!' in exactly 4 words.")
print(f"✅ LLM connected: {response.content}")

def load_entity_data(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    data_path = DATASET_CONFIG["data_path"]
    file_pattern = DATASET_CONFIG["file_pattern"]
    keep_cols = DATASET_CONFIG["keep_columns"]
    entity_col = DATASET_CONFIG["entity_column"]

    entity_datasets = {}
    for entity in state["entities_to_analyze"]:
        file_path = data_path / entity / file_pattern
        if file_path.exists():
            df = pd.read_csv(file_path, usecols=lambda c: c in keep_cols if keep_cols else True)
            df[entity_col] = entity
            entity_datasets[entity] = {
                "entity_name": entity,
                "raw_data": df.to_csv(index=False),
                "cleaned_data": None,
                "row_count": len(df)
            }
    return {"entity_datasets": entity_datasets}


def _is_id_column(df: pd.DataFrame, col: str) -> bool:
    """Detect identifier columns by name patterns AND cardinality."""
    # Name-based: "id", anything ending in "_id", or containing "uuid"/"guid"
    name_lower = col.lower()
    if name_lower == "id" or name_lower.endswith("_id") or "uuid" in name_lower or "guid" in name_lower:
        return True
    # Cardinality-based: numeric column where >90% of values are unique → likely an ID
    if pd.api.types.is_numeric_dtype(df[col]):
        nunique_ratio = df[col].nunique() / max(len(df), 1)
        if nunique_ratio > 0.9:
            return True
    return False


def _detect_date_strings(series: pd.Series, sample_size: int = 50) -> bool:
    """Check if a string column contains parseable date values."""
    sample = series.dropna().head(sample_size).astype(str)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


# --- Detection thresholds (tune these in one place) ---
MISSING_PCT_THRESHOLD = 5        # flag columns with >5% nulls
MISSING_SEVERITY_THRESHOLD = 20  # >20% → high severity
OUTLIER_IQR_FACTOR = 1.5         # standard IQR multiplier
OUTLIER_MIN_COUNT = 10           # minimum outlier rows to report
CURRENCY_SYMBOL_RATIO = 0.5     # >50% of sample has currency symbols
NEAR_CONSTANT_RATIO = 0.95      # >95% one value → near-constant
WHITESPACE_RATIO = 0.05         # >5% of values have leading/trailing spaces
DATE_PARSE_RATIO = 0.8          # >80% of sample parses as dates


def detect_issues(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Detect data quality issues dynamically from the first entity's data."""
    print("\n🔍 Scanning data for issues that need your input...")

    first_entity = list(state["entity_datasets"].keys())[0]
    df = pd.read_csv(pd.io.common.StringIO(state["entity_datasets"][first_entity]["raw_data"]))
    issues = []
    seen_columns = set()  # avoid duplicate issues for the same column

    for col in df.columns:
        if col in seen_columns:
            continue

        # --- Skip identifier columns entirely ---
        if _is_id_column(df, col):
            continue

        # 1. Currency-formatted strings
        if df[col].dtype == object:
            sample = df[col].dropna().head(20).astype(str)
            if len(sample) > 0 and sample.str.contains(r'[$€£]').mean() > CURRENCY_SYMBOL_RATIO:
                issues.append({
                    "issue_type": "currency_format",
                    "column": col,
                    "description": f"Values contain currency symbols (e.g. {sample.iloc[0]})",
                    "examples": sample.head(3).tolist(),
                    "options": ["Convert to numeric (strip currency symbols)", "Keep as text"],
                    "severity": "high"
                })
                seen_columns.add(col)
                continue

            # 2. Boolean t/f encoding
            unique_vals = set(df[col].dropna().unique())
            if unique_vals and unique_vals <= {'t', 'f', 'T', 'F'}:
                issues.append({
                    "issue_type": "boolean_encoding",
                    "column": col,
                    "description": f"Boolean stored as text (values: {sorted(unique_vals)})",
                    "examples": sorted(unique_vals)[:3],
                    "options": ["Convert to True/False", "Convert to 1/0", "Keep as-is"],
                    "severity": "medium"
                })
                seen_columns.add(col)
                continue

            # 3. Date strings stored as text
            if _detect_date_strings(df[col]):
                example_vals = df[col].dropna().head(3).tolist()
                issues.append({
                    "issue_type": "date_format",
                    "column": col,
                    "description": f"Date values stored as text (e.g. {example_vals[0]})",
                    "examples": example_vals,
                    "options": ["Convert to datetime", "Keep as text"],
                    "severity": "medium"
                })
                seen_columns.add(col)
                continue

            # 4. Whitespace issues in string columns
            non_null = df[col].dropna()
            if len(non_null) > 0:
                stripped = non_null.astype(str).str.strip()
                whitespace_count = int((non_null.astype(str) != stripped).sum())
                whitespace_pct = whitespace_count / len(non_null)
                if whitespace_pct > WHITESPACE_RATIO:
                    issues.append({
                        "issue_type": "whitespace",
                        "column": col,
                        "description": f"{whitespace_count:,} values ({whitespace_pct:.1%}) have leading/trailing whitespace",
                        "examples": non_null[non_null.astype(str) != stripped].head(3).tolist(),
                        "options": ["Strip whitespace", "Keep as-is"],
                        "severity": "low"
                    })
                    seen_columns.add(col)
                    # don't continue — may also have missing values

        # 5. Missing values (any column with significant nulls)
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > MISSING_PCT_THRESHOLD:
            missing_count = int(df[col].isnull().sum())

            # Column is entirely (or almost entirely) empty → only offer to drop it
            if missing_pct > 90:
                options = ["Drop column", "Leave empty"]
                severity = "high"
            else:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                options = (
                    ["Fill with median", "Fill with 0", "Drop rows with missing", "Leave empty"]
                    if is_numeric else
                    ["Fill with 'Unknown'", "Fill with most common value", "Drop rows", "Leave empty"]
                )
                severity = "high" if missing_pct > MISSING_SEVERITY_THRESHOLD else "medium"

            issues.append({
                "issue_type": "missing_values",
                "column": col,
                "description": f"{missing_pct:.1f}% missing ({missing_count:,} of {len(df):,} rows)",
                "examples": [f"{missing_count:,} out of {len(df):,} rows"],
                "options": options,
                "severity": severity
            })
            seen_columns.add(col)
            continue

        # 6. Near-constant columns (one value dominates)
        if df[col].nunique() <= 1 and missing_pct <= MISSING_PCT_THRESHOLD:
            dominant = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            issues.append({
                "issue_type": "near_constant",
                "column": col,
                "description": f"Column has only 1 unique value: '{dominant}'",
                "examples": [str(dominant)],
                "options": ["Drop column", "Keep as-is"],
                "severity": "low"
            })
            seen_columns.add(col)
            continue

        # 7. Outliers via IQR (numeric columns only)
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper_fence = q3 + OUTLIER_IQR_FACTOR * iqr
            outlier_count = int((df[col] > upper_fence).sum())
            if iqr > 0 and outlier_count > max(OUTLIER_MIN_COUNT, len(df) * 0.001):
                issues.append({
                    "issue_type": "outliers",
                    "column": col,
                    "description": f"{outlier_count:,} values above upper fence ({upper_fence:,.1f})",
                    "examples": [f"Max: {df[col].max():,.1f}", f"Q3 + 1.5×IQR: {upper_fence:,.1f}"],
                    "options": [f"Cap at {upper_fence:,.1f}", "Remove outlier rows", "Keep all values"],
                    "severity": "medium"
                })
                seen_columns.add(col)

    print(f"   Found {len(issues)} issues that need your preferences")
    print(f"   (Detected from '{first_entity}', will apply to all entities)")
    return {"detected_issues": issues}

def ask_preferences(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Interrupt and ask user for their cleaning preferences."""

    if not state["detected_issues"]:
        print("✅ No issues need preferences — data looks clean!")
        return {"preferences_collected": True, "user_preferences": []}

    num_entities = len(state["entity_datasets"])
    entity_col = state.get("entity_column", "entity")

    print("\n" + "═" * 70)
    print("👤 YOUR INPUT NEEDED — The agent is PAUSED")
    print("═" * 70)
    print(f"\n💡 These preferences will be applied to ALL {num_entities} {entity_col}s!")

    issues_for_display = []
    for i, issue in enumerate(state["detected_issues"]):
        issues_for_display.append({
            "id": i,
            "type": issue["issue_type"],
            "column": issue["column"],
            "description": issue["description"],
            "examples": issue["examples"],
            "severity": issue["severity"],
            "options": {j + 1: opt for j, opt in enumerate(issue["options"])}
        })

    preference_request = {
        "message": "I found data quality issues. How should I handle each one?",
        "note": f"Your choices will be applied consistently to all {num_entities} {entity_col}s!",
        "issues": issues_for_display,
        "total_issues": len(issues_for_display),
        "instructions": "Respond with: {'preferences': [{'issue_id': 0, 'choice': 'your choice'}, ...]}"
    }

    # 🛑 INTERRUPT — Agent PAUSES here!
    user_response = interrupt(preference_request)

    user_preferences = []
    detected_issues = state["detected_issues"]

    for pref in user_response.get("preferences", []):
        issue_id = pref["issue_id"]
        choice = pref["choice"]

        if issue_id >= len(detected_issues):
            continue

        issue = detected_issues[issue_id]
        user_preferences.append({
            "issue_type": issue["issue_type"],
            "column": issue["column"],
            "choice": choice
        })
        print(f"   ✓ {issue['column']}: {choice}")

    print(f"\n   ✅ Received {len(user_preferences)} preferences (will apply to all {entity_col}s!)")

    return {
        "user_preferences": user_preferences,
        "preferences_collected": True
    }


def clean_all_entities(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Apply cleaning preferences to ALL entities."""
    import re
    entity_col = state.get("entity_column", "entity")
    print(f"\n🧹 Applying your preferences to ALL {entity_col}s...")

    updated_datasets = {}
    all_actions = []

    for entity_name, entity_data in state["entity_datasets"].items():
        print(f"\n   Processing {entity_name}...")
        df = pd.read_csv(pd.io.common.StringIO(entity_data["raw_data"]))
        entity_actions = []

        for pref in state["user_preferences"]:
            col = pref["column"]
            choice = pref["choice"]
            issue_type = pref["issue_type"]

            if col not in df.columns:
                continue

            # CURRENCY FORMAT — strip any currency symbol and commas
            if issue_type == "currency_format" and "numeric" in choice.lower():
                df[col] = df[col].astype(str).str.replace(r'[$€£,]', '', regex=True).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                entity_actions.append(f"✓ {entity_name}: Converted '{col}' to numeric")

            # MISSING VALUES
            elif issue_type == "missing_values":
                if "median" in choice.lower():
                    val = df[col].median()
                    df[col] = df[col].fillna(val)
                    entity_actions.append(f"✓ {entity_name}: Filled '{col}' with median ({val:.2f})")
                elif choice.strip() == "0" or "fill with 0" in choice.lower():
                    df[col] = df[col].fillna(0)
                    entity_actions.append(f"✓ {entity_name}: Filled '{col}' with 0")
                elif "unknown" in choice.lower():
                    df[col] = df[col].fillna("Unknown")
                    entity_actions.append(f"✓ {entity_name}: Filled '{col}' with 'Unknown'")
                elif "most common" in choice.lower():
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                        entity_actions.append(f"✓ {entity_name}: Filled '{col}' with most common value")
                elif "drop column" in choice.lower():
                    df = df.drop(columns=[col])
                    entity_actions.append(f"✓ {entity_name}: Dropped empty column '{col}'")
                elif "drop" in choice.lower():
                    before = len(df)
                    df = df.dropna(subset=[col])
                    entity_actions.append(f"✓ {entity_name}: Dropped {before - len(df):,} rows with missing '{col}'")

            # BOOLEAN ENCODING
            elif issue_type == "boolean_encoding":
                if "true/false" in choice.lower():
                    df[col] = df[col].map({'t': True, 'f': False, 'T': True, 'F': False})
                    entity_actions.append(f"✓ {entity_name}: Converted '{col}' to True/False")
                elif "1/0" in choice.lower():
                    df[col] = df[col].map({'t': 1, 'f': 0, 'T': 1, 'F': 0})
                    entity_actions.append(f"✓ {entity_name}: Converted '{col}' to 1/0")

            # OUTLIERS (uses IQR fence for removal)
            elif issue_type == "outliers":
                if "cap" in choice.lower():
                    cap_match = re.search(r'[\d,]+(?:\.\d+)?', choice)
                    if cap_match:
                        cap_val = float(cap_match.group(0).replace(',', ''))
                        df[col] = df[col].clip(upper=cap_val)
                        entity_actions.append(f"✓ {entity_name}: Capped '{col}' at {cap_val:,.1f}")
                elif "remove" in choice.lower():
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    upper_fence = q3 + OUTLIER_IQR_FACTOR * (q3 - q1)
                    before = len(df)
                    df = df[df[col] <= upper_fence]
                    entity_actions.append(f"✓ {entity_name}: Removed {before - len(df):,} outlier rows in '{col}'")

            # DATE FORMAT — convert string dates to datetime
            elif issue_type == "date_format":
                if "convert" in choice.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    entity_actions.append(f"✓ {entity_name}: Converted '{col}' to datetime")

            # WHITESPACE — strip leading/trailing spaces
            elif issue_type == "whitespace":
                if "strip" in choice.lower():
                    df[col] = df[col].astype(str).str.strip()
                    entity_actions.append(f"✓ {entity_name}: Stripped whitespace from '{col}'")

            # NEAR-CONSTANT — drop the column entirely
            elif issue_type == "near_constant":
                if "drop" in choice.lower():
                    df = df.drop(columns=[col])
                    entity_actions.append(f"✓ {entity_name}: Dropped near-constant column '{col}'")

        updated_datasets[entity_name] = {
            **entity_data,
            "cleaned_data": df.to_csv(index=False),
            "row_count": len(df)
        }

        all_actions.extend(entity_actions)
        print(f"      Applied {len(entity_actions)} actions, {len(df):,} rows remaining")

    return {
        "entity_datasets": updated_datasets,
        "cleaning_actions": all_actions
    }


def generate_analysis_prompt(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Use a fast/cheap LLM to write a domain-specific prompt for the analysis step."""

    # Gather real schema info from the cleaned data
    sample_rows = []
    column_profiles = {}

    for entity_name, entity_data in state["entity_datasets"].items():
        df = pd.read_csv(pd.io.common.StringIO(
            entity_data["cleaned_data"] or entity_data["raw_data"]
        ))
        sample_rows = df.head(3).to_dict(orient="records")

        for col in df.columns:
            column_profiles[col] = {
                "dtype": str(df[col].dtype),
                "nulls_pct": round(df[col].isnull().mean() * 100, 1),
                "sample_values": df[col].dropna().head(5).tolist(),
                "unique_count": int(df[col].nunique()),
            }
        break  # profile from first entity; same schema across all

    meta_prompt = f"""
You are a prompt engineer. Your task is to write a precise LLM prompt
that will instruct another LLM to choose the best analysis for this dataset.

DATASET SCHEMA:
{json.dumps(column_profiles, indent=2)}

SAMPLE ROWS:
{json.dumps(sample_rows, indent=2)}

ENTITIES AVAILABLE: {list(state["entity_datasets"].keys())}

Write a prompt (plain text, no JSON) that:
1. Describes what this dataset appears to contain (infer the domain)
2. Lists the most analytically interesting columns and why
3. Asks the LLM to pick one analysis type and return a JSON task spec
   with fields: task_name, task_description, required_columns,
   analysis_type, group_by_column
Keep the prompt under 400 words.
"""

    # Use a cheaper/faster model for prompt generation
    prompt_llm = ChatBedrockConverse(
        model="us.anthropic.claude-haiku-3-20240307-v1:0",  # fast + cheap
        client=bedrock_client,
    )

    response = llm.invoke(meta_prompt)
    generated_prompt = response.content

    print(f"\n🧠 Generated analysis prompt ({len(generated_prompt)} chars)")

    return {"generated_analysis_prompt": generated_prompt}


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


def decide_analysis_task(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Let the LLM propose multiple analysis options for the user to choose from."""
    print("\n🧠 LLM is proposing analysis options...")

    entity_col = state.get("entity_column", "entity")
    entities_info = []
    all_columns = set()

    for entity_name, entity_data in state["entity_datasets"].items():
        df = pd.read_csv(pd.io.common.StringIO(entity_data["cleaned_data"] or entity_data["raw_data"]))
        summary = {entity_col: entity_name, "rows": len(df)}
        for col in df.select_dtypes(include="number").columns:
            summary[f"avg_{col}"] = round(df[col].mean(), 2)
        entities_info.append(summary)
        all_columns.update(df.columns.tolist())

    # Use the meta-generated prompt if available, else fall back to a generic one
    base_prompt = state.get("generated_analysis_prompt") or f"""
You are a data analyst. You have a dataset partitioned by '{entity_col}'.
Entities: {[e[entity_col] for e in entities_info]}
Available columns: {list(all_columns)}
Data summary:
{json.dumps(entities_info, indent=2)}

Propose 3 DIFFERENT analysis tasks that would provide the most valuable insights.
"""

    full_prompt = base_prompt + """

Propose 3 DIFFERENT analysis tasks, ranked by value. Each must focus on a different angle.

Return as a JSON ARRAY of 3 options (best first, inside a ```json block):
[
  {
    "task_name": "Short name for the analysis",
    "task_description": "What exactly you will analyze and why it's valuable",
    "required_columns": ["col1", "col2"],
    "analysis_type": "comparison" | "ranking" | "trends" | "segmentation" | "distribution",
    "group_by_column": null
  },
  ...
]
"""

    response = llm.invoke(full_prompt)

    try:
        parsed = _parse_json_from_response(response.content)
        if isinstance(parsed, list):
            analysis_options = parsed[:3]
        else:
            analysis_options = [parsed]
    except Exception:
        analysis_options = [{
            "task_name": "Cross-Entity Comparison",
            "task_description": "Compare key metrics across all entities",
            "required_columns": list(all_columns)[:5],
            "analysis_type": "comparison",
            "group_by_column": None
        }]

    print(f"\n   📊 LLM proposed {len(analysis_options)} analysis options:")
    for i, opt in enumerate(analysis_options, 1):
        print(f"\n   Option {i}: {opt['task_name']}")
        print(f"      {opt['task_description']}")

    return {"analysis_options": analysis_options}


def choose_analysis_task(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Interrupt and let the user pick one of the proposed analysis options."""
    analysis_options = state.get("analysis_options", [])

    if not analysis_options:
        print("⚠️ No analysis options available — skipping choice.")
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

    print(f"\n   ✅ You picked Option {choice}: {chosen['task_name']}")
    print(f"   📝 {chosen['task_description']}")
    return {"analysis_task": chosen}


def execute_analysis(state: MultiDatasetAnalysisState) -> MultiDatasetAnalysisState:
    """Execute the LLM-decided analysis across all entities."""
    print(f"\n🔬 Executing: {state['analysis_task']['task_name']}...")

    entity_col = state.get("entity_column", "entity")
    task = state['analysis_task']

    # Combine all entity data
    all_dfs = []
    for entity_name, entity_data in state["entity_datasets"].items():
        df = pd.read_csv(pd.io.common.StringIO(entity_data["cleaned_data"] or entity_data["raw_data"]))
        df[entity_col] = entity_name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Dynamic aggregation over all numeric columns
    numeric_cols = [c for c in combined_df.select_dtypes(include="number").columns if c != entity_col]

    # Count rows per entity separately (avoids collision if first_col is numeric)
    record_counts = combined_df.groupby(entity_col).size().rename("records")

    if numeric_cols:
        agg_dict = {col: ['mean', 'median'] for col in numeric_cols}
        column_names = []
        for col in numeric_cols:
            column_names.extend([f'avg_{col}', f'median_{col}'])

        entity_stats = combined_df.groupby(entity_col).agg(agg_dict).round(2)
        entity_stats.columns = column_names
        entity_stats.insert(0, 'records', record_counts)
    else:
        entity_stats = record_counts.to_frame()

    # Optional group-by breakdown (LLM-suggested)
    group_col = task.get("group_by_column")
    group_breakdown_str = "N/A"
    group_breakdown_dict = {}

    if group_col and group_col in combined_df.columns:
        group_breakdown = combined_df.groupby([entity_col, group_col]).size().unstack(fill_value=0)
        group_breakdown_str = group_breakdown.to_string()
        group_breakdown_dict = group_breakdown.to_dict()

    analysis_prompt = f"""
You are executing the analysis: "{task['task_name']}"
Task: {task['task_description']}

DATA ANALYSIS RESULTS:

1. ENTITY-LEVEL STATISTICS (grouped by '{entity_col}'):
{entity_stats.to_string()}

2. BREAKDOWN BY '{group_col or "N/A"}':
{group_breakdown_str}

Based on this data, provide:
1. 5-7 specific, actionable insights that answer the analysis goal
2. A 2-3 sentence executive summary

Format as JSON:
{{
    "insights": ["insight 1", "insight 2", ...],
    "summary": "Executive summary here"
}}
"""

    response = llm.invoke(analysis_prompt)

    try:
        content = response.content
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    result = json.loads(part)
                    break
                except Exception:
                    continue
            else:
                result = json.loads(content)
        else:
            result = json.loads(content)
        insights = result.get("insights", [])
        summary = result.get("summary", "Analysis complete.")
    except Exception:
        insights = [response.content]
        summary = "Analysis complete."

    print(f"   Generated {len(insights)} insights")
    return {
        "analysis_results": {
            "entity_stats": entity_stats.to_dict(),
            "group_breakdown": group_breakdown_dict,
            "total_records": len(combined_df)
        },
        "insights": insights,
        "summary": summary
    }

