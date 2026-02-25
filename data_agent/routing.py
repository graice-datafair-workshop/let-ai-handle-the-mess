from typing import Literal

from .states import CombinedAgentState


def route_after_inference(state: CombinedAgentState) -> Literal[
    'ask_feedback', 'apply_transforms', 'infer_type'
]:
    """
    Router: Decide whether to ask for feedback, continue inferring, or apply transformations.
    Works with multi-file state structure.
    """
    current_file_idx = state.get('current_file_index', 0)
    files_to_process = state['files_to_process']

    if current_file_idx >= len(files_to_process):
        return 'apply_transforms'

    file_path = files_to_process[current_file_idx]
    file_state = state['file_states'][file_path]

    current_col_idx = file_state['current_column_index']
    total_columns = len(file_state['column_names'])

    if current_col_idx >= total_columns:
        return 'apply_transforms'

    if state['status'] == 'awaiting_feedback':
        return 'ask_feedback'
    else:
        return 'infer_type'


def route_after_nil_values(
    state: CombinedAgentState
) -> Literal['handle_nil_values', 'show_statistics']:
    """
    Router: After handling one nil column, either continue to next nil column or show statistics.
    """
    if state['status'] == 'checking_nil_values':
        return 'handle_nil_values'
    else:
        return 'show_statistics'


def route_after_statistics(
    state: CombinedAgentState
) -> Literal['load_data', 'generate_analysis_prompt']:
    """
    Router: After showing statistics and column drop, either load next file or start analysis.
    """
    if state['status'] == 'complete':
        return 'generate_analysis_prompt'
    else:
        return 'load_data'
