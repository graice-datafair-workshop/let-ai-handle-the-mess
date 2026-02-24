from langgraph.graph import StateGraph, START, END

from .states import CombinedAgentState
from .cleaning_nodes import (
    load_file_data,
    infer_current_column_type,
    ask_user_feedback,
    apply_transformations,
    check_duplicates,
    handle_nil_values,
    show_column_statistics,
)
from .analysis_nodes import (
    generate_analysis_prompt,
    decide_analysis_task,
    choose_analysis_task,
    execute_analysis,
)
from .visualization_nodes import (
    decide_visualization,
    choose_visualization,
    create_visualization,
    export_results,
)
from .routing import (
    route_after_inference,
    route_after_nil_values,
    route_after_statistics,
)


def build_graph():
    builder = StateGraph(CombinedAgentState)

    # =========================================================================
    # DATA CLEANING PHASE
    # =========================================================================
    builder.add_node("load_data", load_file_data)
    builder.add_node("infer_type", infer_current_column_type)
    builder.add_node("ask_feedback", ask_user_feedback)
    builder.add_node("apply_transforms", apply_transformations)
    builder.add_node("check_duplicates", check_duplicates)
    builder.add_node("handle_nil_values", handle_nil_values)
    builder.add_node("show_statistics", show_column_statistics)

    # =========================================================================
    # ANALYSIS & VISUALIZATION PHASE
    # =========================================================================
    builder.add_node("generate_analysis_prompt", generate_analysis_prompt)
    builder.add_node("decide_analysis", decide_analysis_task)
    builder.add_node("choose_analysis", choose_analysis_task)
    builder.add_node("execute_analysis", execute_analysis)
    builder.add_node("decide_visualization", decide_visualization)
    builder.add_node("choose_visualization", choose_visualization)
    builder.add_node("create_visualization", create_visualization)
    builder.add_node("export_results", export_results)

    # =========================================================================
    # FLOW: DATA CLEANING
    # =========================================================================
    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "infer_type")

    builder.add_conditional_edges(
        "infer_type",
        route_after_inference,
        {
            'ask_feedback': 'ask_feedback',
            'infer_type': 'infer_type',
            'apply_transforms': 'apply_transforms'
        }
    )

    builder.add_conditional_edges(
        "ask_feedback",
        route_after_inference,
        {
            'ask_feedback': 'ask_feedback',
            'infer_type': 'infer_type',
            'apply_transforms': 'apply_transforms'
        }
    )

    builder.add_edge("apply_transforms", "check_duplicates")
    builder.add_edge("check_duplicates", "handle_nil_values")

    builder.add_conditional_edges(
        "handle_nil_values",
        route_after_nil_values,
        {
            'handle_nil_values': 'handle_nil_values',
            'show_statistics': 'show_statistics'
        }
    )

    builder.add_conditional_edges(
        "show_statistics",
        route_after_statistics,
        {
            'load_data': 'load_data',
            'generate_analysis_prompt': 'generate_analysis_prompt'
        }
    )

    # =========================================================================
    # FLOW: ANALYSIS & VISUALIZATION
    # =========================================================================
    builder.add_edge("generate_analysis_prompt", "decide_analysis")
    builder.add_edge("decide_analysis", "choose_analysis")
    builder.add_edge("choose_analysis", "execute_analysis")
    builder.add_edge("execute_analysis", "decide_visualization")
    builder.add_edge("decide_visualization", "choose_visualization")
    builder.add_edge("choose_visualization", "create_visualization")
    builder.add_edge("create_visualization", "export_results")
    builder.add_edge("export_results", END)

    return builder
