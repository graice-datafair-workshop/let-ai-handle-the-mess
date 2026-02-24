from pathlib import Path

from langgraph.types import interrupt

from .states import CombinedAgentState, FileState, ColumnInfo
from .helper_functions import (
    load_file_as_dataframe,
    infer_column_type,
    get_llm_type_inference,
    interpret_user_response,
    apply_column_transformation,
    detect_duplicates,
    interpret_duplicate_decision,
    analyze_nil_values,
    apply_nil_value_handling,
    interpret_nil_value_decision,
    compute_column_statistics,
    interpret_column_drop_decision,
    interpret_apply_to_all_decision,
)


# =============================================================================
# NODE: LOAD FILE DATA
# =============================================================================

def load_file_data(state: CombinedAgentState) -> dict:
    """
    Node: Load the current file (CSV or PKL) into the state.
    Supports multi-file processing - loads one file at a time.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']

    if current_file_idx >= len(files_to_process):
        print("All files processed!")
        return {'status': 'complete'}

    file_path = files_to_process[current_file_idx]
    print(f"\n{'='*60}")
    print(f"FILE {current_file_idx + 1}/{len(files_to_process)}: {file_path}")
    print(f"{'='*60}")

    try:
        df = load_file_as_dataframe(file_path)
        column_names = df.columns.tolist()

        print(f"Loaded {len(df)} rows × {len(column_names)} columns")

        new_file_state: FileState = {
            'file_path': file_path,
            'original_df': df.copy(),
            'working_df': df.copy(),
            'column_names': column_names,
            'current_column_index': 0,
            'column_info': {},
            'status': 'inferring'
        }

        updated_file_states = dict(state.get('file_states', {}))
        updated_file_states[file_path] = new_file_state

        return {
            'file_states': updated_file_states,
            'status': 'inferring',
            'processing_log': [
                f"[File {current_file_idx + 1}] Loaded {len(df)} rows × {len(column_names)} columns from {file_path}"
            ]
        }
    except Exception as e:
        print(f"Error loading file: {e}")
        return {
            'status': 'error',
            'processing_log': [f"ERROR loading file {file_path}: {str(e)}"]
        }


# =============================================================================
# NODE: INFER CURRENT COLUMN TYPE
# =============================================================================

def infer_current_column_type(state: CombinedAgentState) -> dict:
    """
    Node: Infer the type of the current column in the current file.
    Uses both rule-based inference and LLM-based inference for enhanced accuracy.
    If apply_to_all is set and we have stored decisions, skip inference entirely.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']
    apply_to_all = state.get('apply_decisions_to_all_files', False)

    if current_file_idx >= len(files_to_process):
        print("All files processed!")
        return {'status': 'complete'}

    file_path = files_to_process[current_file_idx]
    file_state = state['file_states'][file_path]

    column_names = file_state['column_names']
    current_col_idx = file_state['current_column_index']

    if current_col_idx >= len(column_names):
        print(f"All columns in '{file_path}' processed!")
        updated_file_states = dict(state['file_states'])
        updated_file = dict(updated_file_states[file_path])
        updated_file['status'] = 'transforming'
        updated_file_states[file_path] = FileState(**updated_file)
        return {
            'file_states': updated_file_states,
            'status': 'transforming'
        }

    column_name = column_names[current_col_idx]

    # Check if we should auto-apply stored decisions (skip inference + LLM entirely)
    # Apply if: (apply_to_all AND not first file) OR (decisions were pre-loaded)
    decisions_preloaded = state.get('decisions_preloaded', False)
    should_auto_apply = (apply_to_all and current_file_idx > 0) or decisions_preloaded
    
    if should_auto_apply:
        stored_column_decisions = state.get('stored_column_decisions', {})
        if column_name in stored_column_decisions:
            stored_decision = stored_column_decisions[column_name]
            print(f"\n[File {current_file_idx + 1}] Auto-applying stored decision for column {current_col_idx + 1}/{len(column_names)}: '{column_name}'")
            print(f"   Type: {stored_decision.get('type')}")
            if stored_decision.get('correction'):
                print(f"   Correction: {stored_decision.get('correction')}")

            # Build column info from stored decision
            col_info = {
                'column_name': column_name,
                'inferred_type': stored_decision.get('type', 'text'),
                'reasoning': 'Auto-applied from first file decisions.',
                'sample_values': [],
                'confidence': 1.0,
                'user_approved': stored_decision.get('approved', True),
                'user_correction': stored_decision.get('correction'),
                'transformation_applied': False
            }

            # Update file state and move to next column
            updated_file_states = dict(state['file_states'])
            updated_file = dict(updated_file_states[file_path])
            updated_column_info = dict(updated_file.get('column_info', {}))
            updated_column_info[column_name] = ColumnInfo(**col_info)
            updated_file['column_info'] = updated_column_info
            updated_file['current_column_index'] = current_col_idx + 1
            updated_file_states[file_path] = FileState(**updated_file)

            return {
                'file_states': updated_file_states,
                'status': 'inferring'  # Continue to next column
            }

    # Normal inference path (first file or column not in stored decisions)
    print(f"\n🔍 [File {current_file_idx + 1}] Analyzing column {current_col_idx + 1}/{len(column_names)}: '{column_name}'")

    df = file_state['working_df']

    # Step 1: Rule-based inference
    col_info = infer_column_type(df[column_name], column_name)
    print(f"   Rule-based inference: {col_info['inferred_type']} (confidence: {col_info['confidence']:.1%})")

    # Step 2: LLM-based inference
    print(f"   Consulting LLM for additional analysis...")
    llm_suggestion = get_llm_type_inference(
        column_name=column_name,
        sample_values=col_info['sample_values'],
        rule_based_type=col_info['inferred_type'],
        rule_based_reasoning=col_info['reasoning'],
        rule_based_confidence=col_info['confidence']
    )

    # Combine insights
    if llm_suggestion.suggested_type != col_info['inferred_type']:
        print(f"   LLM disagrees: {llm_suggestion.suggested_type} (LLM confidence: {llm_suggestion.confidence:.1%})")
        col_info['confidence'] = 0.5
        col_info['reasoning'] = f"Disagreement: Rule-based: {col_info['inferred_type']}. LLM suggests: {llm_suggestion.suggested_type} - {llm_suggestion.reasoning}"
    else:
        print(f"   ✓ LLM agrees: {llm_suggestion.suggested_type}")
        col_info['reasoning'] = f"{col_info['reasoning']} LLM confirms: {llm_suggestion.reasoning}"
        col_info['confidence'] = llm_suggestion.confidence

    needs_feedback = col_info['confidence'] < 0.8

    if needs_feedback:
        print(f"   Confidence {col_info['confidence']:.1%} < 80% - will ask user for feedback")
    else:
        print(f"   Confidence {col_info['confidence']:.1%} >= 80% - auto-approving")
        col_info['user_approved'] = True

    updated_file_states = dict(state['file_states'])
    updated_file = dict(updated_file_states[file_path])
    updated_column_info = dict(updated_file.get('column_info', {}))
    updated_column_info[column_name] = ColumnInfo(**col_info)
    updated_file['column_info'] = updated_column_info
    updated_file_states[file_path] = FileState(**updated_file)

    if needs_feedback:
        return {
            'file_states': updated_file_states,
            'status': 'awaiting_feedback'
        }
    else:
        updated_file['current_column_index'] = current_col_idx + 1
        updated_file_states[file_path] = FileState(**updated_file)
        return {
            'file_states': updated_file_states,
            'status': 'inferring'
        }


# =============================================================================
# NODE: ASK USER FEEDBACK
# =============================================================================

def ask_user_feedback(state: CombinedAgentState) -> dict:
    """
    Node: Present the inference to the user and ask for feedback.
    Uses LangGraph's interrupt() to pause and wait for user input.
    Uses LLM to interpret user responses. Auto-applies stored decisions for subsequent files.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']
    file_path = files_to_process[current_file_idx]
    file_state = state['file_states'][file_path]
    apply_to_all = state.get('apply_decisions_to_all_files', False)

    column_names = file_state['column_names']
    current_col_idx = file_state['current_column_index']
    column_name = column_names[current_col_idx]
    col_info = file_state['column_info'][column_name]

    # Check if we should auto-apply stored column decision
    # Apply if: (apply_to_all AND not first file) OR (decisions were pre-loaded)
    decisions_preloaded = state.get('decisions_preloaded', False)
    auto_apply = False
    stored_decision = None
    if (apply_to_all and current_file_idx > 0) or decisions_preloaded:
        stored_column_decisions = state.get('stored_column_decisions', {})
        if column_name in stored_column_decisions:
            stored_decision = stored_column_decisions[column_name]
            auto_apply = True
            print(f"\nAuto-applying stored decision for column '{column_name}':")
            print(f"   Type: {stored_decision.get('type')}")
            print(f"   Approved: {stored_decision.get('approved')}")

    if not auto_apply:
        num_samples = state.get('_show_samples_count', 5)
        samples_to_show = col_info['sample_values'][:num_samples]

        file_name = Path(file_path).name
        inferred_type_upper = col_info['inferred_type'].upper()
        confidence_pct = f"{col_info['confidence']:.1%}"

        message = (
            f"\n╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  FILE {current_file_idx + 1}/{len(files_to_process)}: {file_name}\n"
            f"║  COLUMN {current_col_idx + 1} of {len(column_names)}: {column_name}\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Inferred Type: {inferred_type_upper}\n"
            f"║  Confidence: {confidence_pct}\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Reasoning:\n"
            f"║  {col_info['reasoning']}\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Sample Values ({len(samples_to_show)} shown):\n"
            f"║  {samples_to_show}\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  OPTIONS: approve / correct to <type> / skip / show more samples\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n"
        )
        print(message)

        # INTERRUPT: Wait for user input
        user_response = interrupt({
            'file_path': file_path,
            'column_name': column_name,
            'inferred_type': col_info['inferred_type'],
            'confidence': col_info['confidence'],
            'prompt': f"What type should '{column_name}' be? (or approve '{col_info['inferred_type']}')"
        })

        print(f"\n🤖 Interpreting your response: '{user_response}'...")
        interpretation = interpret_user_response(
            str(user_response), column_name, col_info['inferred_type'], col_info['sample_values']
        )

        print(f"   Action: {interpretation.action}")
        print(f"   Reasoning: {interpretation.reasoning}")

        if interpretation.action == 'show_more_samples':
            new_count = min(num_samples + 5, len(col_info['sample_values']))
            print(f"\nShowing {new_count} samples...")
            return {'_show_samples_count': new_count, 'status': 'awaiting_feedback'}

        user_approved = interpretation.action == 'approve'
        user_correction = None
        if interpretation.action == 'correct' and interpretation.target_type:
            user_correction = interpretation.target_type
        elif interpretation.action == 'skip':
            user_correction = 'skip'
    else:
        user_approved = stored_decision.get('approved', False)
        user_correction = stored_decision.get('correction')

    updated_file_states = dict(state['file_states'])
    updated_file = dict(updated_file_states[file_path])
    updated_column_info = dict(updated_file['column_info'])
    updated_col = dict(updated_column_info[column_name])
    updated_col['user_approved'] = user_approved
    updated_col['user_correction'] = user_correction
    updated_column_info[column_name] = ColumnInfo(**updated_col)
    updated_file['column_info'] = updated_column_info
    updated_file['current_column_index'] = current_col_idx + 1
    updated_file_states[file_path] = FileState(**updated_file)

    return {
        'file_states': updated_file_states,
        'status': 'inferring',
        '_show_samples_count': 5
    }


# =============================================================================
# NODE: APPLY TRANSFORMATIONS
# =============================================================================

def apply_transformations(state: CombinedAgentState) -> dict:
    """
    Node: Apply all approved/corrected transformations to the current file's DataFrame.
    After transforming, moves to data cleaning phase.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']
    file_path = files_to_process[current_file_idx]
    file_state = state['file_states'][file_path]

    print(f"\n🔧 Applying transformations to file {current_file_idx + 1}/{len(files_to_process)}: {Path(file_path).name}")

    df = file_state['working_df'].copy()
    log_messages = []

    updated_file_states = dict(state['file_states'])
    updated_file = dict(updated_file_states[file_path])
    updated_column_info = dict(updated_file['column_info'])

    for column_name, col_info in file_state['column_info'].items():
        if col_info['user_correction'] == 'skip':
            log_messages.append(f"Skipped '{column_name}' (user requested)")
            continue

        if col_info['user_correction'] is not None:
            target_type = col_info['user_correction']
        elif col_info['user_approved']:
            target_type = col_info['inferred_type']
        else:
            continue

        df, log_msg = apply_column_transformation(df, column_name, target_type)
        log_messages.append(log_msg)

        updated_col = dict(updated_column_info[column_name])
        updated_col['transformation_applied'] = True
        updated_column_info[column_name] = ColumnInfo(**updated_col)

    print(f"\nTransformation Summary for {Path(file_path).name}:")
    for msg in log_messages:
        print(f"   • {msg}")

    updated_file['working_df'] = df
    updated_file['column_info'] = updated_column_info
    updated_file['status'] = 'transformed'
    updated_file_states[file_path] = FileState(**updated_file)

    # Save checkpoint: transformed file
    original_path = Path(file_path)
    stem = original_path.stem
    transformed_output_name = f"{stem}_transformed.pkl"
    transformed_output_path = original_path.parent / transformed_output_name
    df.to_pickle(transformed_output_path)
    print(f"\nSaved transformed data checkpoint to: {transformed_output_path}")

    print(f"\nTransformations applied! Moving to data cleaning phase...")
    return {
        'file_states': updated_file_states,
        'status': 'checking_duplicates',
        'processing_log': [f"[File {current_file_idx + 1}] Transformations applied: " + "; ".join(log_messages)]
    }


# =============================================================================
# NODE: CHECK DUPLICATES
# =============================================================================

def check_duplicates(state: CombinedAgentState) -> dict:
    """
    Node: Check for duplicate rows and ask user if they want to drop them.
    Uses LLM to interpret user response. Auto-applies stored decision for subsequent files.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']
    file_path = files_to_process[current_file_idx]
    file_state = state['file_states'][file_path]
    df = file_state['working_df']
    apply_to_all = state.get('apply_decisions_to_all_files', False)

    print(f"\n{'='*60}")
    print(f"DATA CLEANING: Checking for Duplicate Rows")
    print(f"{'='*60}")

    duplicate_count, duplicate_rows = detect_duplicates(df)

    if duplicate_count == 0:
        print("No duplicate rows found!")
        return {
            'status': 'checking_nil_values',
            'stored_duplicate_decision': state.get('stored_duplicate_decision'),
            'processing_log': [f"[File {current_file_idx + 1}] No duplicate rows found"]
        }

    dup_pct = (duplicate_count / len(df)) * 100
    print(f"\nFound {duplicate_count} duplicate rows ({dup_pct:.2f}% of data)")
    print(f"\nSample duplicate rows:")
    print(duplicate_rows.head(3).to_string())

    # Check if we should auto-apply stored decision
    # Apply if: (apply_to_all AND not first file) OR (decisions were pre-loaded)
    decisions_preloaded = state.get('decisions_preloaded', False)
    drop_duplicates = False
    if (apply_to_all and current_file_idx > 0) or decisions_preloaded:
        stored_decision = state.get('stored_duplicate_decision')
        if stored_decision is not None:
            drop_duplicates = stored_decision
            action_str = 'Drop duplicates' if drop_duplicates else 'Keep duplicates'
            print(f"\nAuto-applying stored decision: {action_str}")
        else:
            apply_to_all = False

    if not (apply_to_all and current_file_idx > 0):
        message = (
            f"\n╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  DUPLICATE ROWS DETECTED\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Found: {duplicate_count} duplicate rows ({dup_pct:.2f}%)\n"
            f"║  Would you like to drop these duplicate rows?\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n"
        )
        print(message)

        user_response = interrupt({
            'type': 'duplicate_check',
            'duplicate_count': duplicate_count,
            'prompt': f"Found {duplicate_count} duplicate rows. Drop them?"
        })

        print(f"\nInterpreting your response: '{user_response}'...")
        decision = interpret_duplicate_decision(str(user_response), duplicate_count)
        action_str = 'Drop duplicates' if decision.drop_duplicates else 'Keep duplicates'
        print(f"   Decision: {action_str}")
        print(f"   Reasoning: {decision.reasoning}")
        drop_duplicates = decision.drop_duplicates

    updated_file_states = dict(state['file_states'])
    updated_file = dict(updated_file_states[file_path])

    if drop_duplicates:
        df_cleaned = df.drop_duplicates(keep='first')
        updated_file['working_df'] = df_cleaned
        log_msg = f"Dropped {duplicate_count} duplicate rows"
        print(f"\n{log_msg}. New shape: {df_cleaned.shape}")
    else:
        log_msg = "Kept duplicate rows"
        print(f"\n{log_msg}")

    updated_file_states[file_path] = FileState(**updated_file)

    return {
        'file_states': updated_file_states,
        'status': 'checking_nil_values',
        'stored_duplicate_decision': drop_duplicates,
        'processing_log': [f"[File {current_file_idx + 1}] {log_msg}"]
    }


# =============================================================================
# NODE: HANDLE NIL VALUES
# =============================================================================

def handle_nil_values(state: CombinedAgentState) -> dict:
    """
    Node: Check for nil values and ask user what to do for each column with nil values.
    Uses LLM to interpret user responses. Auto-applies stored decisions for subsequent files.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']
    file_path = files_to_process[current_file_idx]
    file_state = state['file_states'][file_path]
    df = file_state['working_df'].copy()
    apply_to_all = state.get('apply_decisions_to_all_files', False)

    # Track which nil column we're processing
    nil_columns_processed = state.get('_nil_columns_processed', [])
    stored_nil_decisions = dict(state.get('stored_nil_decisions', {}))

    print(f"\n{'='*60}")
    print(f"DATA CLEANING: Checking for Nil/Missing Values")
    print(f"{'='*60}")

    nil_analysis = analyze_nil_values(df)

    if not nil_analysis:
        print("No missing values found in any column!")
        return {
            'status': 'showing_statistics',
            '_nil_columns_processed': [],
            'stored_nil_decisions': stored_nil_decisions,
            'processing_log': [f"[File {current_file_idx + 1}] No missing values found"]
        }

    # Find the next column to process
    remaining_columns = [
        col for col in nil_analysis.keys() if col not in nil_columns_processed
    ]

    if not remaining_columns:
        print("All columns with nil values have been processed!")
        return {
            'status': 'showing_statistics',
            '_nil_columns_processed': [],
            'stored_nil_decisions': stored_nil_decisions,
            'processing_log': [
                f"[File {current_file_idx + 1}] Nil value handling complete"
            ]
        }

    # Process the next column
    column_name = remaining_columns[0]
    col_info = nil_analysis[column_name]
    is_numeric = col_info['is_numeric']

    print(f"\nColumn with missing values: {column_name}")
    print(f"   Missing: {col_info['nil_count']} values ({col_info['nil_percentage']}%)")
    print(f"   Type: {'Numeric' if is_numeric else 'Text/Categorical'}")

    # Check if we should auto-apply stored decision
    # Apply if: (apply_to_all AND not first file) OR (decisions were pre-loaded)
    decisions_preloaded = state.get('decisions_preloaded', False)
    action = None
    if (apply_to_all and current_file_idx > 0) or decisions_preloaded:
        stored_action = stored_nil_decisions.get(column_name)
        if stored_action:
            action = stored_action
            print(f"\nAuto-applying stored decision: {action}")

    if action is None:
        if is_numeric:
            options = [
                "Fill with median", "Fill with 0",
                "Drop rows with missing", "Leave empty"
            ]
        else:
            options = [
                "Fill with 'Unknown'", "Fill with most common",
                "Drop rows", "Leave empty"
            ]

        message = (
            f"\n╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  MISSING VALUES IN: {column_name}\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Missing: {col_info['nil_count']} ({col_info['nil_percentage']}%)\n"
            f"║  Column type: {'Numeric' if is_numeric else 'Text/Categorical'}\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Options:\n"
        )
        for opt in options:
            message += f"║    • {opt}\n"
        message += (
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  How would you like to handle missing values in '{column_name}'?\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n"
        )
        print(message)

        # INTERRUPT: Wait for user input
        user_response = interrupt({
            'type': 'nil_value_handling',
            'column_name': column_name,
            'nil_count': col_info['nil_count'],
            'nil_percentage': col_info['nil_percentage'],
            'is_numeric': is_numeric,
            'prompt': f"How to handle nil values in '{column_name}'?"
        })

        print(f"\n Interpreting your response: '{user_response}'...")
        decision = interpret_nil_value_decision(
            str(user_response), column_name, is_numeric, col_info['nil_count']
        )
        print(f"   Action: {decision.action}")
        print(f"   Reasoning: {decision.reasoning}")
        action = decision.action

    # Store this decision
    stored_nil_decisions[column_name] = action

    # Apply the nil value handling
    df, log_msg = apply_nil_value_handling(df, column_name, action)
    print(f"\n {log_msg}")

    # Update file state
    updated_file_states = dict(state['file_states'])
    updated_file = dict(updated_file_states[file_path])
    updated_file['working_df'] = df
    updated_file_states[file_path] = FileState(**updated_file)

    updated_nil_columns_processed = nil_columns_processed + [column_name]

    remaining_after = [
        col for col in nil_analysis.keys() if col not in updated_nil_columns_processed
    ]

    return {
        'file_states': updated_file_states,
        'status': 'checking_nil_values' if remaining_after else 'showing_statistics',
        '_nil_columns_processed': updated_nil_columns_processed,
        'stored_nil_decisions': stored_nil_decisions,
        'processing_log': [f"[File {current_file_idx + 1}] {log_msg}"]
    }


# =============================================================================
# NODE: SHOW COLUMN STATISTICS
# =============================================================================

def show_column_statistics(state: CombinedAgentState) -> dict:
    """
    Node: Show column statistics including outliers, then ask if user wants to drop columns.
    Uses LLM to interpret user response about which columns to drop.
    After first file, asks if user wants to apply same decisions to remaining files.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']
    file_path = files_to_process[current_file_idx]
    file_state = state['file_states'][file_path]
    df = file_state['working_df']
    apply_to_all = state.get('apply_decisions_to_all_files', False)

    print(f"\n{'='*60}")
    print(f"DATA EXPLORATION: Column Statistics")
    print(f"{'='*60}")

    stats = compute_column_statistics(df)

    # Build a nice summary table
    header = (
        f"{'Column':<25} {'Type':<12} {'Non-Null':<10} {'Unique':<8} "
        f"{'Min':<12} {'Max':<12} {'Median':<12} {'Outliers':<10}"
    )
    print(f"\n{header}")
    print("-" * 110)

    for s in stats:
        col_name = s['column_name'][:24]
        dtype = str(s['dtype'])[:11]
        non_null = str(s['non_null_count'])
        unique = str(s['unique_count'])

        if 'min' in s:
            min_val = f"{s['min']:.2f}" if abs(s['min']) < 1e6 else f"{s['min']:.2e}"
            max_val = f"{s['max']:.2f}" if abs(s['max']) < 1e6 else f"{s['max']:.2e}"
            med_val = f"{s['median']:.2f}" if abs(s['median']) < 1e6 else f"{s['median']:.2e}"
            outliers = f"{s['outlier_count']} " if s['has_outliers'] else "0"
        else:
            min_val = max_val = med_val = "N/A"
            outliers = "N/A"

        row = (
            f"{col_name:<25} {dtype:<12} {non_null:<10} {unique:<8} "
            f"{min_val:<12} {max_val:<12} {med_val:<12} {outliers:<10}"
        )
        print(row)

    available_columns = [s['column_name'] for s in stats]

    # Check if we should auto-apply stored decision
    # Apply if: (apply_to_all AND not first file) OR (decisions were pre-loaded)
    decisions_preloaded = state.get('decisions_preloaded', False)
    columns_to_drop = []
    if (apply_to_all and current_file_idx > 0) or decisions_preloaded:
        stored_cols_to_drop = state.get('stored_columns_to_drop', [])
        columns_to_drop = [c for c in stored_cols_to_drop if c in available_columns]
        if columns_to_drop:
            print(f"\n Auto-applying stored decision: dropping columns {columns_to_drop}")
        else:
            print(f"\n Auto-applying stored decision: no columns to drop")
    else:
        message = (
            f"\n╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  COLUMN DROP DECISION\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Review the statistics above.\n"
            f"║  Would you like to drop any columns?\n"
            f"║  \n"
            f"║  Examples:\n"
            f"║  • 'drop column1, column2' - drop specific columns\n"
            f"║  • 'remove the id column' - drop by name\n"
            f"║  • 'no' / 'none' / 'skip' - don't drop anything\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n"
        )
        print(message)

        user_response = interrupt({
            'type': 'column_drop',
            'available_columns': available_columns,
            'statistics': stats,
            'prompt': "Which columns would you like to drop? (or 'none')"
        })

        print(f"\nInterpreting your response: '{user_response}'...")
        decision = interpret_column_drop_decision(str(user_response), available_columns)
        cols = decision.columns_to_drop if decision.columns_to_drop else 'None'
        print(f"   Columns to drop: {cols}")
        print(f"   Reasoning: {decision.reasoning}")
        columns_to_drop = decision.columns_to_drop

    updated_file_states = dict(state['file_states'])
    updated_file = dict(updated_file_states[file_path])

    if columns_to_drop:
        valid_cols = [c for c in columns_to_drop if c in df.columns]
        if valid_cols:
            df_cleaned = df.drop(columns=valid_cols)
            updated_file['working_df'] = df_cleaned
            log_msg = f"Dropped columns: {valid_cols}"
            print(f"\n{log_msg}. New shape: {df_cleaned.shape}")
        else:
            log_msg = "No valid columns to drop"
            print(f"\n{log_msg}")
            valid_cols = []
    else:
        log_msg = "No columns dropped"
        print(f"\n {log_msg}")
        valid_cols = []

    updated_file_states[file_path] = FileState(**updated_file)

    # Save the cleaned DataFrame
    original_path = Path(file_path)
    stem = original_path.stem
    output_name = f"{stem}_cleaned.pkl"
    output_path = original_path.parent / output_name

    updated_file['working_df'].to_pickle(output_path)
    print(f"\nSaved cleaned data to: {output_path}")

    next_file_idx = current_file_idx + 1
    remaining_files = len(files_to_process) - next_file_idx

    # Continued in Part 2
    return _show_column_statistics_part2(
        state, updated_file_states, file_state, valid_cols, log_msg, output_path,
        current_file_idx, next_file_idx, remaining_files, files_to_process
    )


# =============================================================================
# NODE: SHOW COLUMN STATISTICS (Part 2 - Apply to All Logic)
# =============================================================================

def _show_column_statistics_part2(
    state, updated_file_states, file_state, valid_cols, log_msg, output_path,
    current_file_idx, next_file_idx, remaining_files, files_to_process
):
    """Helper function for show_column_statistics - handles apply-to-all logic."""
    
    if next_file_idx < len(files_to_process):
        # First file and there are more files - ask about applying decisions
        if current_file_idx == 0 and remaining_files > 0:
            stored_column_decisions = {}
            for col_name, col_info in file_state['column_info'].items():
                stored_column_decisions[col_name] = {
                    'type': col_info.get('user_correction') or col_info.get('inferred_type'),
                    'approved': col_info.get('user_approved', False),
                    'correction': col_info.get('user_correction')
                }

            stored_nil_decisions = state.get('stored_nil_decisions', {})
            stored_duplicate_decision = state.get('stored_duplicate_decision')

            print(f"\n{'='*60}")
            print(f"APPLY DECISIONS TO REMAINING FILES?")
            print(f"{'='*60}")
            print(f"\nYou have {remaining_files} more file(s) to process.")
            print(f"\nDecisions made for this file:")
            print(f"   • Column type conversions: {len(stored_column_decisions)} columns")
            dup_action = 'Drop' if stored_duplicate_decision else 'Keep'
            print(f"   • Duplicate handling: {dup_action}")
            print(f"   • Nil value handling: {len(stored_nil_decisions)} columns")
            drop_cols = valid_cols if valid_cols else 'None'
            print(f"   • Columns to drop: {drop_cols}")

            apply_message = (
                f"\n╔══════════════════════════════════════════════════════════════════╗\n"
                f"║  APPLY SAME DECISIONS TO REMAINING FILES?\n"
                f"╠══════════════════════════════════════════════════════════════════╣\n"
                f"║  Would you like to apply these same decisions to the\n"
                f"║  remaining {remaining_files} file(s)?\n"
                f"║  \n"
                f"║  • 'yes' / 'apply to all' — Use same decisions for all files\n"
                f"║  • 'no' / 'manually' — Make decisions for each file separately\n"
                f"╚══════════════════════════════════════════════════════════════════╝\n"
            )
            print(apply_message)

            apply_response = interrupt({
                'type': 'apply_to_all',
                'remaining_files': remaining_files,
                'prompt': f"Apply same decisions to remaining {remaining_files} file(s)?"
            })

            print(f"\nInterpreting your response: '{apply_response}'...")
            apply_decision = interpret_apply_to_all_decision(
                str(apply_response), remaining_files
            )
            action = 'Apply to all files' if apply_decision.apply_to_all else 'Process manually'
            print(f"   Decision: {action}")
            print(f"   Reasoning: {apply_decision.reasoning}")

            print(f"\nFile {current_file_idx + 1} complete! Moving to file {next_file_idx + 1}...")

            return {
                'file_states': updated_file_states,
                'current_file_index': next_file_idx,
                'status': 'loading',
                'apply_decisions_to_all_files': apply_decision.apply_to_all,
                'stored_column_decisions': stored_column_decisions,
                'stored_duplicate_decision': stored_duplicate_decision,
                'stored_nil_decisions': stored_nil_decisions,
                'stored_columns_to_drop': valid_cols,
                'processing_log': [
                    f"[File {current_file_idx + 1}] {log_msg}. Saved to {output_path}. "
                    f"Apply to all: {apply_decision.apply_to_all}"
                ]
            }
        else:
            # Subsequent files - just continue
            print(f"\nFile {current_file_idx + 1} complete! Moving to file {next_file_idx + 1}...")
            return {
                'file_states': updated_file_states,
                'current_file_index': next_file_idx,
                'status': 'loading',
                'processing_log': [
                    f"[File {current_file_idx + 1}] {log_msg}. Saved to {output_path}"
                ]
            }
    else:
        print(f"\nAll {len(files_to_process)} file(s) processed and cleaned!")
        return {
            'file_states': updated_file_states,
            'status': 'complete',
            'processing_log': [
                f"[File {current_file_idx + 1}] {log_msg}. Saved to {output_path}"
            ]
        }
