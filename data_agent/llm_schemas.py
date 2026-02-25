from typing import Literal

from pydantic import BaseModel, Field

from .llm_setup import llm


class UserResponseInterpretation(BaseModel):
    """Structured interpretation of user's response about a column."""
    action: Literal["approve", "correct", "skip", "show_more_samples"] = Field(
        description="The action the user wants to take. "
                    "'approve' = accept the inferred type, "
                    "'correct' = change to a different type, "
                    "'skip' = leave column unchanged, "
                    "'show_more_samples' = user wants to see more sample values"
    )
    target_type: str | None = Field(
        default=None,
        description="If action is 'correct', the type the user wants. "
                    "Valid types: integer, float, boolean, datetime, categorical, text, identifier"
    )
    reasoning: str = Field(
        description="Brief explanation of how you interpreted the user's response"
    )


class DuplicateDecision(BaseModel):
    """Structured interpretation of user's response about duplicate rows."""
    drop_duplicates: bool = Field(
        description="Whether the user wants to drop duplicate rows. True = drop, False = keep."
    )
    reasoning: str = Field(
        description="Brief explanation of how you interpreted the user's response"
    )


class NilValueDecision(BaseModel):
    """Structured interpretation of user's response about handling nil/missing values."""
    action: Literal["fill_median", "fill_zero", "fill_unknown", "fill_most_common", "drop_rows", "leave_empty"] = Field(
        description="The action the user wants to take for handling nil values."
    )
    reasoning: str = Field(
        description="Brief explanation of how you interpreted the user's response"
    )


class ColumnDropDecision(BaseModel):
    """Structured interpretation of user's response about dropping columns."""
    columns_to_drop: list[str] = Field(
        default=[],
        description="List of column names the user wants to drop. Empty list if no columns should be dropped."
    )
    reasoning: str = Field(
        description="Brief explanation of how you interpreted the user's response"
    )


class ApplyToAllDecision(BaseModel):
    """Structured interpretation of user's response about applying decisions to all remaining files."""
    apply_to_all: bool = Field(
        description="Whether the user wants to apply the same decisions to all remaining files."
    )
    reasoning: str = Field(
        description="Brief explanation of how you interpreted the user's response"
    )


class LLMTypeInference(BaseModel):
    """LLM's analysis of a column's data type."""
    suggested_type: Literal[
        "integer", "float", "boolean", "datetime", "categorical", "text", "identifier"
    ] = Field(
        description="The data type the LLM suggests for this column. "
                    "Valid types: integer, float, boolean, datetime, categorical, text, identifier"
    )
    confidence: float = Field(
        description="Confidence level between 0.0 and 1.0 for this type suggestion",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this type was chosen based on column name and sample values"
    )


# Create structured LLMs for each schema
structured_llm = llm.with_structured_output(UserResponseInterpretation)
duplicate_decision_llm = llm.with_structured_output(DuplicateDecision)
nil_value_decision_llm = llm.with_structured_output(NilValueDecision)
column_drop_decision_llm = llm.with_structured_output(ColumnDropDecision)
apply_to_all_decision_llm = llm.with_structured_output(ApplyToAllDecision)
type_inference_llm = llm.with_structured_output(LLMTypeInference)
