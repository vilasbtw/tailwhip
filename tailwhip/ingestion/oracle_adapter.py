import pandas as pd
from pathlib import Path

from tailwhip.models.column_metadata import ColumnMetadata


class OracleAdapter:
    """
    Reads an Oracle-exported CSV (sep=';', encoding='latin-1') and returns
    a flat list of ColumnMetadata — one entry per column.

    Expected CSV format:
        OWNER;TABLE_NAME;TABLE_COMMENT;COLUMN_NAME;DATA_TYPE;COLUMN_COMMENT;CONSTRAINT_TYPE

    CONSTRAINT_TYPE values:
        'PK'                     -> is_pk=True
        'FK -> OWNER.TABLE_NAME' -> is_fk=True, fk_ref_schema and fk_ref_table populated
        empty / NaN              -> no constraint
    """

    def parse(self, filepath: Path) -> list[ColumnMetadata]:
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, sep=";", encoding="latin-1", dtype=str)
        df = df.fillna("")

        columns = []
        for _, row in df.iterrows():
            is_pk, is_fk, fk_ref_schema, fk_ref_table = self._parse_constraint(
                row.get("CONSTRAINT_TYPE", "")
            )

            # TABLE_COMMENT is repeated on every row of the same table;
            # used as fallback description to enrich the embedding text
            table_comment = row.get("TABLE_COMMENT", "").strip() or None
            col_comment = row.get("COLUMN_COMMENT", "").strip() or None

            col = ColumnMetadata(
                table_name=row["TABLE_NAME"].strip(),
                schema_name=row["OWNER"].strip() or None,
                column_name=row["COLUMN_NAME"].strip(),
                data_type=row["DATA_TYPE"].strip(),
                description=col_comment or table_comment,
                is_pk=is_pk,
                is_fk=is_fk,
                fk_ref_schema=fk_ref_schema,
                fk_ref_table=fk_ref_table,
            )
            columns.append(col)

        return columns

    def _parse_constraint(
        self, constraint: str
    ) -> tuple[bool, bool, str | None, str | None]:
        """
        Parse the CONSTRAINT_TYPE field and return
        (is_pk, is_fk, fk_ref_schema, fk_ref_table).
        """
        constraint = constraint.strip()

        if constraint == "PK":
            return True, False, None, None

        if constraint.startswith("FK -> "):
            ref = constraint.removeprefix("FK -> ").strip()
            parts = ref.split(".", 1)
            if len(parts) == 2:
                return False, True, parts[0], parts[1]
            # no owner in the reference
            return False, True, None, parts[0]

        return False, False, None, None
