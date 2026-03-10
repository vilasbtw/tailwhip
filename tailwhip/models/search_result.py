from pydantic import BaseModel, field_validator

from tailwhip.models.table_metadata import TableMetadata
from tailwhip.models.column_metadata import ColumnMetadata


class SearchResult(BaseModel):
    """A single ranked result returned by the search pipeline."""

    rank: int
    doc_id: str
    table: TableMetadata
    relevant_columns: list[ColumnMetadata]
    final_score: float

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rank must be >= 1")
        return v

    @field_validator("final_score")
    @classmethod
    def validate_final_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("final_score must be between 0 and 1")
        return v
