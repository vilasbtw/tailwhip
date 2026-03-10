from tailwhip.models.table_metadata import TableMetadata
from tailwhip.models.column_metadata import ColumnMetadata
from tailwhip.models.schema_document import SchemaDocument


class DocumentBuilder:
    """
    Builds SchemaDocument instances from TableMetadata and ColumnMetadata.

    Responsibilities:
    - Generate doc_id in the format "SCHEMA.TABLE_NAME" (or just "TABLE_NAME" if no schema)
    - Delegate embedding text assembly to SchemaDocument.to_embedding_text()
    - Support batch building via build_all()
    """

    def build(self, table: TableMetadata, columns: list[ColumnMetadata]) -> SchemaDocument:
        doc_id = (
            f"{table.schema_name}.{table.table_name}"
            if table.schema_name
            else table.table_name
        )

        return SchemaDocument(
            doc_id=doc_id,
            table=table,
            columns=columns,
        )

    def build_all(
        self, pairs: list[tuple[TableMetadata, list[ColumnMetadata]]]
    ) -> list[SchemaDocument]:
        return [self.build(table, columns) for table, columns in pairs]
