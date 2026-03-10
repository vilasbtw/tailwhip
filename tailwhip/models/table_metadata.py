from pydantic import BaseModel, field_validator


class TableMetadata(BaseModel):
    """Metadata for a single Oracle table."""

    table_name: str
    schema_name: str | None = None
    description: str | None = None
    column_count: int = 0
    fk_in_count: int = 0

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("table_name cannot be empty")
        return v

    @field_validator("fk_in_count")
    @classmethod
    def validate_fk_in_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("fk_in_count cannot be negative")
        return v

    @field_validator("column_count")
    @classmethod
    def validate_column_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("column_count cannot be negative")
        return v
