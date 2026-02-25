import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .states import ColumnInfo
from .llm_schemas import (
    LLMTypeInference,
    UserResponseInterpretation,
    DuplicateDecision,
    NilValueDecision,
    ColumnDropDecision,
    ApplyToAllDecision,
    type_inference_llm,
    structured_llm,
    duplicate_decision_llm,
    nil_value_decision_llm,
    column_drop_decision_llm,
    apply_to_all_decision_llm,
)


# =============================================================================
# FILE LABELLING
# =============================================================================

def unique_file_label(file_path: str, all_file_paths: list[str] | None = None) -> str:
    """
    Return a human-readable label for a file path.
    If multiple files share the same stem (e.g. all called 'listings.csv'),
    use the parent directory name instead (e.g. 'Barcelona', 'Athens').
    """
    stem = Path(file_path).stem
    if all_file_paths is not None:
        stems = [Path(fp).stem for fp in all_file_paths]
        if stems.count(stem) > 1:
            return Path(file_path).parent.name
    return stem


# =============================================================================
# FILE LOADING
# =============================================================================

def load_file_as_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a file (CSV or PKL) as a DataFrame.
    All columns are read as strings to avoid automatic type inference.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    elif suffix in ['.pkl', '.pickle']:
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Pickle file does not contain a DataFrame: {type(df)}")
        df = df.astype(str)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .csv, .pkl, .pickle")

    return df


# =============================================================================
# TYPE INFERENCE
# =============================================================================

def infer_column_type(series: pd.Series, column_name: str) -> ColumnInfo:
    """
    Infer the data type of a column with detailed reasoning.
    Only returns these 7 basic pandas-compatible types: text, boolean, identifier, integer, float, datetime, categorical
    """
    non_empty = series[series != '']
    total_count = len(series)
    non_empty_count = len(non_empty)
    unique_count = series.nunique()
    unique_ratio = unique_count / total_count if total_count > 0 else 0
    sample_values = non_empty.head(5).tolist() if len(non_empty) > 0 else []

    # Handle empty column
    if non_empty_count == 0:
        return ColumnInfo(
            column_name=column_name, inferred_type='text',
            reasoning=f'All {total_count} values are empty or missing. Defaulting to text.',
            sample_values=[], confidence=1.0,
            user_approved=None, user_correction=None, transformation_applied=False
        )

    # 1. Check for identifier
    col_lower = column_name.lower()
    if unique_ratio > 0.95 and any(x in col_lower for x in ['id', '_id', 'identifier', 'key']):
        return ColumnInfo(
            column_name=column_name, inferred_type='identifier',
            reasoning=f'Column name suggests ID, high uniqueness: {unique_ratio:.1%}.',
            sample_values=sample_values, confidence=0.9,
            user_approved=None, user_correction=None, transformation_applied=False
        )

    # 2. Check for boolean
    unique_lower = set(non_empty.str.lower().str.strip().unique())
    boolean_sets = [{'t', 'f'}, {'true', 'false'}, {'yes', 'no'}, {'y', 'n'}, {'0', '1'}]
    for bool_set in boolean_sets:
        if unique_lower.issubset(bool_set) and len(unique_lower) <= 2:
            return ColumnInfo(
                column_name=column_name, inferred_type='boolean',
                reasoning=f'Contains only boolean-like values: {unique_lower}.',
                sample_values=sample_values, confidence=0.95,
                user_approved=None, user_correction=None, transformation_applied=False
            )

    # 3. Check for numeric types
    numeric_values = pd.to_numeric(non_empty, errors='coerce')
    numeric_ratio = numeric_values.notna().sum() / len(non_empty) if len(non_empty) > 0 else 0

    if numeric_ratio > 0.9:
        # Filter out NaN and infinite values to avoid IntCastingNaNError
        valid_nums = numeric_values.dropna()
        valid_nums = valid_nums[np.isfinite(valid_nums)]
        # Check if all values are integers (no decimal part)
        is_integer = len(valid_nums) > 0 and (valid_nums == valid_nums.round()).all()
        if is_integer:
            return ColumnInfo(
                column_name=column_name, inferred_type='integer',
                reasoning=f'{numeric_ratio:.1%} are valid integers.',
                sample_values=sample_values, confidence=numeric_ratio,
                user_approved=None, user_correction=None, transformation_applied=False
            )
        else:
            return ColumnInfo(
                column_name=column_name, inferred_type='float',
                reasoning=f'{numeric_ratio:.1%} are valid numbers with decimals.',
                sample_values=sample_values, confidence=numeric_ratio,
                user_approved=None, user_correction=None, transformation_applied=False
            )

    # 4. Check for datetime
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Could not infer format')
            date_values = pd.to_datetime(non_empty.head(50), errors='coerce', format='mixed')
        date_ratio = date_values.notna().sum() / min(len(non_empty), 50)
        if date_ratio > 0.8:
            return ColumnInfo(
                column_name=column_name, inferred_type='datetime',
                reasoning=f'{date_ratio:.1%} of sample values parse as dates.',
                sample_values=sample_values, confidence=date_ratio,
                user_approved=None, user_correction=None, transformation_applied=False
            )
    except Exception:
        pass

    # 5. Check for categorical (low cardinality)
    if unique_ratio < 0.05 and unique_count <= 20:
        return ColumnInfo(
            column_name=column_name, inferred_type='categorical',
            reasoning=f'Low cardinality: {unique_count} unique values ({unique_ratio:.1%}).',
            sample_values=sample_values, confidence=0.85,
            user_approved=None, user_correction=None, transformation_applied=False
        )

    # Default to text
    return ColumnInfo(
        column_name=column_name, inferred_type='text',
        reasoning=f'High cardinality text: {unique_count} unique values.',
        sample_values=sample_values, confidence=0.7,
        user_approved=None, user_correction=None, transformation_applied=False
    )


# =============================================================================
# DATA TRANSFORMATION
# =============================================================================

def apply_column_transformation(df: pd.DataFrame, column_name: str, target_type: str) -> tuple[pd.DataFrame, str]:
    """Apply a transformation to convert a column to the specified type."""
    if column_name not in df.columns:
        return df, f"ERROR: Column '{column_name}' not found"

    try:
        if target_type == 'integer':
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')
            return df, f"Converted '{column_name}' to integer (nullable Int64)"

        elif target_type == 'float':
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            return df, f"Converted '{column_name}' to float"

        elif target_type == 'boolean':
            bool_map = {
                't': True, 'f': False, 'true': True, 'false': False,
                'yes': True, 'no': False, 'y': True, 'n': False,
                '1': True, '0': False,
            }
            df[column_name] = df[column_name].str.lower().str.strip().map(bool_map)
            return df, f"Converted '{column_name}' to boolean"

        elif target_type == 'datetime':
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format='mixed')
            return df, f"Converted '{column_name}' to datetime"

        elif target_type in ('categorical', 'category'):
            df[column_name] = df[column_name].astype('category')
            return df, f"Converted '{column_name}' to categorical"

        elif target_type == 'currency':
            cleaned = df[column_name].str.replace(r'[$€£¥₹,]', '', regex=True)
            df[column_name] = pd.to_numeric(cleaned, errors='coerce')
            return df, f"Converted '{column_name}' to numeric (currency symbols removed)"

        elif target_type in ('text', 'string'):
            df[column_name] = df[column_name].astype(str)
            return df, f"Converted '{column_name}' to string"

        elif target_type == 'identifier':
            df[column_name] = df[column_name].astype(str)
            return df, f"Kept '{column_name}' as identifier (string)"

        else:
            return df, f"Unknown type '{target_type}' - no transformation applied"

    except Exception as e:
        return df, f"ERROR transforming '{column_name}': {str(e)}"


# =============================================================================
# DATA CLEANING HELPERS
# =============================================================================

def detect_duplicates(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Detect duplicate rows in a DataFrame."""
    duplicates = df[df.duplicated(keep='first')]
    return len(duplicates), duplicates


def analyze_nil_values(df: pd.DataFrame) -> dict[str, dict]:
    """Analyze nil/missing values in each column."""
    nil_analysis = {}
    for col in df.columns:
        nil_mask = df[col].isna() | (df[col].astype(str).str.strip() == '') | (df[col].astype(str).str.lower() == 'nan')
        nil_count = nil_mask.sum()
        nil_percentage = (nil_count / len(df)) * 100 if len(df) > 0 else 0
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        if nil_count > 0:
            nil_analysis[col] = {
                'nil_count': int(nil_count),
                'nil_percentage': round(nil_percentage, 2),
                'is_numeric': is_numeric
            }
    return nil_analysis


def apply_nil_value_handling(df: pd.DataFrame, column_name: str, action: str) -> tuple[pd.DataFrame, str]:
    """Apply the selected nil value handling action to a column."""
    if column_name not in df.columns:
        return df, f"ERROR: Column '{column_name}' not found"

    try:
        nil_mask = df[column_name].isna() | (df[column_name].astype(str).str.strip() == '') | (df[column_name].astype(str).str.lower() == 'nan')
        nil_count = nil_mask.sum()

        if action == 'fill_median':
            median_val = df[column_name].median()
            df.loc[nil_mask, column_name] = median_val
            return df, f"Filled {nil_count} nil values in '{column_name}' with median ({median_val:.2f})"

        elif action == 'fill_zero':
            df.loc[nil_mask, column_name] = 0
            return df, f"Filled {nil_count} nil values in '{column_name}' with 0"

        elif action == 'fill_unknown':
            df.loc[nil_mask, column_name] = 'Unknown'
            return df, f"Filled {nil_count} nil values in '{column_name}' with 'Unknown'"

        elif action == 'fill_most_common':
            most_common = df[column_name].mode().iloc[0] if len(df[column_name].mode()) > 0 else 'Unknown'
            df.loc[nil_mask, column_name] = most_common
            return df, f"Filled {nil_count} nil values in '{column_name}' with most common value ('{most_common}')"

        elif action == 'drop_rows':
            df = df[~nil_mask].reset_index(drop=True)
            return df, f"Dropped {nil_count} rows with nil values in '{column_name}'"

        elif action == 'leave_empty':
            return df, f"Left {nil_count} nil values in '{column_name}' unchanged"

        else:
            return df, f"Unknown action '{action}' for nil value handling"

    except Exception as e:
        return df, f"ERROR handling nil values in '{column_name}': {str(e)}"


def compute_column_statistics(df: pd.DataFrame) -> list[dict]:
    """Compute statistics for each column including outlier detection."""
    stats = []
    for col in df.columns:
        col_stats = {
            'column_name': col,
            'dtype': str(df[col].dtype),
            'non_null_count': int(df[col].notna().sum()),
            'null_count': int(df[col].isna().sum()),
            'unique_count': int(df[col].nunique())
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_stats['min'] = float(col_data.min())
                col_stats['max'] = float(col_data.max())
                col_stats['median'] = float(col_data.median())
                col_stats['mean'] = float(col_data.mean())
                col_stats['std'] = float(col_data.std())

                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                col_stats['outlier_count'] = len(outliers)
                col_stats['has_outliers'] = len(outliers) > 0
        else:
            col_stats['has_outliers'] = False
            col_stats['outlier_count'] = 0

        stats.append(col_stats)
    return stats


# =============================================================================
# LLM INTERPRETATION FUNCTIONS
# =============================================================================

def get_llm_type_inference(
    column_name: str,
    sample_values: list[str],
    rule_based_type: str,
    rule_based_reasoning: str,
    rule_based_confidence: float
) -> LLMTypeInference:
    """Use the LLM to infer the data type of a column based on its name and sample values."""
    prompt = f"""You are a data type inference expert. \
Analyze the following column and determine its most appropriate data type.

Column Information:
- Column name: {column_name}
- Sample values: {sample_values[:10]}

Rule-based Analysis (for context):
- Inferred type: {rule_based_type}
- Reasoning: {rule_based_reasoning}
- Confidence: {rule_based_confidence:.1%}

Valid data types (ONLY these 7 types):
- integer: Whole numbers
- float: Decimal numbers (including monetary values)
- boolean: True/False values
- datetime: Date or datetime values
- categorical: Limited set of categories
- text: Free-form text
- identifier: Unique identifiers (IDs, keys)
"""
    return type_inference_llm.invoke(prompt)


def interpret_user_response(
    user_input: str,
    column_name: str,
    inferred_type: str,
    sample_values: list[str]
) -> UserResponseInterpretation:
    """Use the LLM to interpret what the user wants to do with this column."""
    prompt = f"""You are helping interpret a user's response about a data column type inference.

Context:
- Column name: {column_name}
- Our inferred type: {inferred_type}
- Sample values: {sample_values[:5]}

The user was asked if they agree with the inferred type '{inferred_type}' for column '{column_name}'.

User's response: "{user_input}"

Interpret what the user wants:
- If they agree/approve/confirm, action is 'approve'
- If they want a different type (e.g., "make it a date", "this should be categorical"), action is 'correct'
- If they want to skip/ignore/leave as-is without converting, action is 'skip'
- If they want to see more data/samples/examples, action is 'show_more_samples'

Valid target types for 'correct' action: integer, float, boolean, datetime, categorical, text, identifier
"""
    return structured_llm.invoke(prompt)


def interpret_duplicate_decision(user_input: str, duplicate_count: int) -> DuplicateDecision:
    """Use LLM to interpret if user wants to drop duplicate rows."""
    prompt = f"""You are helping interpret a user's response about handling duplicate rows in a dataset.

Context:
- The dataset has {duplicate_count} duplicate rows.
- The user was asked if they want to drop these duplicates.

User's response: "{user_input}"

Interpret what the user wants:
- If they agree/approve/yes/drop/remove, they want to drop duplicates
- If they decline/no/keep/skip, they want to keep duplicates
"""
    return duplicate_decision_llm.invoke(prompt)


def interpret_nil_value_decision(user_input: str, column_name: str, is_numeric: bool, nil_count: int) -> NilValueDecision:
    """Use LLM to interpret how user wants to handle nil values in a column."""
    options = "'fill_median', 'fill_zero', 'drop_rows', 'leave_empty'" if is_numeric else "'fill_unknown', 'fill_most_common', 'drop_rows', 'leave_empty'"

    prompt = f"""You are helping interpret a user's response about handling missing/nil values in a column.

Context:
- Column name: {column_name}
- Column type: {'numeric' if is_numeric else 'text/categorical'}
- Number of missing values: {nil_count}
- Available options: {options}

User's response: "{user_input}"

Interpret what the user wants:
- 'fill_median' = fill with median value (numeric columns)
- 'fill_zero' = fill with 0 (numeric columns)
- 'fill_unknown' = fill with 'Unknown' string (text columns)
- 'fill_most_common' = fill with most common/frequent value (text columns)
- 'drop_rows' = drop all rows with missing values in this column
- 'leave_empty' = don't change anything, keep nil values
"""
    return nil_value_decision_llm.invoke(prompt)


def interpret_column_drop_decision(user_input: str, available_columns: list[str]) -> ColumnDropDecision:
    """Use LLM to interpret which columns user wants to drop."""
    prompt = f"""You are helping interpret a user's response about dropping columns from a dataset.

Context:
- Available columns: {available_columns}
- The user was shown column statistics and asked which columns they want to drop.

User's response: "{user_input}"

Interpret what the user wants:
- Extract the column names they want to drop
- If they say 'none', 'no', 'skip', or similar, return an empty list
- Match column names flexibly (partial matches, case-insensitive)
- Only include columns that exist in the available_columns list
"""
    return column_drop_decision_llm.invoke(prompt)


def interpret_apply_to_all_decision(user_input: str, remaining_files_count: int) -> ApplyToAllDecision:
    """Use LLM to interpret if user wants to apply the same decisions to all remaining files."""
    prompt = f"""You are helping interpret a user's response about applying data cleaning decisions to multiple files.

Context:
- The first file has been processed with various data cleaning decisions.
- There are {remaining_files_count} more file(s) to process.
- The user was asked if they want to apply the same decisions to all remaining files.

User's response: "{user_input}"

Interpret what the user wants:
- If they agree/yes/apply/same/reuse/all, they want to apply the same decisions to all files
- If they decline/no/different/manually/each/individual, they want to make decisions for each file separately
"""
    return apply_to_all_decision_llm.invoke(prompt)


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


def _make_json_serializable(obj):
    """Convert non-JSON-serializable objects (like Timestamps) to strings."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime, Timestamp, etc.
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj


# =============================================================================
# DECISION STORAGE AND REUSE
# =============================================================================

def load_saved_decisions(decisions_file: str, existing_files: list) -> tuple:

    if Path(decisions_file).exists():
        with open(decisions_file, 'r') as f:
            decisions = json.load(f)

        print(f"Loaded decisions from: {decisions_file}")
        print(f"  Column type decisions: {len(decisions['stored_column_decisions'])} columns")
        dup_action = 'Drop' if decisions['stored_duplicate_decision'] else 'Keep'
        print(f"  Duplicate handling: {dup_action}")
        print(f"  Nil value handling: {len(decisions['stored_nil_decisions'])} columns")
        print(f"  Columns to drop: {decisions['stored_columns_to_drop'] or 'None'}")

        initial_state = {
            'files_to_process': existing_files,
            'current_file_index': 0,
            'status': 'initializing',
            'file_states': {},
            'processing_log': ['Loaded saved decisions - will auto-apply to all files'],
            'apply_decisions_to_all_files': True,
            'decisions_preloaded': True,
            'stored_column_decisions': decisions['stored_column_decisions'],
            'stored_duplicate_decision': decisions['stored_duplicate_decision'],
            'stored_nil_decisions': decisions['stored_nil_decisions'],
            'stored_columns_to_drop': decisions['stored_columns_to_drop'],
            'analysis_options': [],
            'analysis_task': {},
            'analysis_results': {},
            'insights': [],
            'summary': '',
            'visualization_options': [],
            'visualization_task': {},
            'visualization_figure': None,
        }

        config = {'configurable': {'thread_id': 'loaded_decisions_run_1'}}
        print("\nInitial state created with loaded decisions!")
        return initial_state, config
    else:
        print(f"No decisions file found ({decisions_file}). Using fresh initial state.")
        return None, None