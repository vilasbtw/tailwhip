from pydantic import BaseModel, field_validator


class ColumnMetadata(BaseModel):
    """Metadata for a single Oracle column."""

    table_name: str
    schema_name: str | None = None
    column_name: str
    data_type: str
    description: str | None = None
    is_pk: bool = False
    is_fk: bool = False
    fk_ref_schema: str | None = None
    fk_ref_table: str | None = None

    @field_validator("column_name")
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("column_name cannot be empty")
        return v

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("data_type cannot be empty")
        return v
