from collections import defaultdict

from tailwhip.models.column_metadata import ColumnMetadata
from tailwhip.models.table_metadata import TableMetadata


class SchemaNormalizer:
    """
    Takes a flat list of ColumnMetadata (one per column) and returns
    a list of TableMetadata with column_count and fk_in_count computed.

    fk_in_count: number of foreign keys from *other* tables pointing to
    this table. Self-references are excluded.
    """

    def normalize(self, columns: list[ColumnMetadata]) -> list[TableMetadata]:
        if not columns:
            return []

        # Group columns by (schema_name, table_name)
        groups: dict[tuple, list[ColumnMetadata]] = defaultdict(list)
        for col in columns:
            key = (col.schema_name, col.table_name)
            groups[key].append(col)

        # Count inbound FKs: for each FK column, increment the target table's counter
        fk_in_count: dict[str, int] = defaultdict(int)
        for col in columns:
            if col.is_fk and col.fk_ref_table:
                if col.fk_ref_table == col.table_name:
                    continue  # skip self-references
                fk_in_count[col.fk_ref_table] += 1

        # Build TableMetadata for each group
        tables = []
        for (schema_name, table_name), cols in groups.items():
            table = TableMetadata(
                table_name=table_name,
                schema_name=schema_name,
                column_count=len(cols),
                fk_in_count=fk_in_count.get(table_name, 0),
            )
            tables.append(table)

        return tables
