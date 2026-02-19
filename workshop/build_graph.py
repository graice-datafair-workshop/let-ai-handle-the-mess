from langgraph.graph import StateGraph, START, END
from states import MultiDatasetAnalysisState
import nodes
import visualization

def build_graph():
    builder = StateGraph(MultiDatasetAnalysisState)

    # Add all nodes
    builder.add_node("load_data", nodes.load_entity_data)
    builder.add_node("detect_issues", nodes.detect_issues)
    builder.add_node("ask_preferences", nodes.ask_preferences)
    builder.add_node("clean_all_entities", nodes.clean_all_entities)
    builder.add_node("generate_analysis_prompt", nodes.generate_analysis_prompt)
    builder.add_node("decide_analysis", nodes.decide_analysis_task)
    builder.add_node("choose_analysis", nodes.choose_analysis_task)
    builder.add_node("execute_analysis", nodes.execute_analysis)
    builder.add_node("decide_visualization", visualization.decide_visualization)
    builder.add_node("choose_visualization", visualization.choose_visualization)
    builder.add_node("create_visualization", visualization.create_visualization)
    builder.add_node("export_results", visualization.export_results)

    # Define the flow
    builder.add_edge(START, "load_data")
    builder.add_edge("load_data", "detect_issues")
    builder.add_edge("detect_issues", "ask_preferences")
    builder.add_edge("ask_preferences", "clean_all_entities")
    builder.add_edge("clean_all_entities", "generate_analysis_prompt")  # meta-prompt first
    builder.add_edge("generate_analysis_prompt", "decide_analysis")
    builder.add_edge("decide_analysis", "choose_analysis")             # INTERRUPT: user picks analysis
    builder.add_edge("choose_analysis", "execute_analysis")
    builder.add_edge("execute_analysis", "decide_visualization")
    builder.add_edge("decide_visualization", "choose_visualization")   # INTERRUPT: user picks a viz
    builder.add_edge("choose_visualization", "create_visualization")
    builder.add_edge("create_visualization", "export_results")
    builder.add_edge("export_results", END)

    return builder

