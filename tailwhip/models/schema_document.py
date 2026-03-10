from pydantic import BaseModel, field_validator

from tailwhip.models.table_metadata import TableMetadata
from tailwhip.models.column_metadata import ColumnMetadata


def tokenize_name(name: str) -> str:
    """
    Convert an Oracle identifier into readable tokens for embedding.

        "SYS_USER_ACCESS" -> "sys user access"
        "COD_USER"        -> "cod users"
        "USERS"           -> "users"
    """
    return " ".join(name.lower().split("_"))


class SchemaDocument(BaseModel):
    """
    A searchable unit representing a single Oracle table.

    Combines table metadata and all its columns into a text
    suitable for vector embedding and BM25 indexing.
    """

    doc_id: str  # format: "SCHEMA.TABLE_NAME"
    table: TableMetadata
    columns: list[ColumnMetadata]

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v

    def to_embedding_text(self) -> str:
        """
        Build a rich text representation for embedding.

        Tokenizes table and column names — critical because many tables
        have no comments. Without tokenization, "SYS_USERS" and
        "SYS_USER_DATA" would produce nearly identical vectors.

        Example output for SYS_USER with no comments:
            Table: sys user | Schema: MYSCHEMA
            Columns: id user (NUMBER) [PK],
                     name (VARCHAR2),
                     email (VARCHAR2),
                     active (CHAR),
                     id system (NUMBER) [FK -> SYS_SYSTEM]
            Referenced by 5 tables
        """
        table_token = tokenize_name(self.table.table_name)
        schema = self.table.schema_name or ""

        header = f"Table: {table_token} | Schema: {schema}"

        if self.table.description:
            header += f"\nDescription: {self.table.description}"

        col_parts = []
        for col in self.columns:
            col_token = tokenize_name(col.column_name)
            part = f"{col_token} ({col.data_type})"

            if col.description:
                part += f" - {col.description}"

            if col.is_pk:
                part += " [PK]"
            elif col.is_fk and col.fk_ref_table:
                ref = (
                    f"{col.fk_ref_schema}.{col.fk_ref_table}"
                    if col.fk_ref_schema
                    else col.fk_ref_table
                )
                part += f" [FK -> {ref}]"

            col_parts.append(part)

        body = "Columns: " + ",\n          ".join(col_parts)

        parts = [header, body]

        if self.table.fk_in_count > 0:
            parts.append(f"Referenced by {self.table.fk_in_count} tables")

        return "\n".join(parts)
