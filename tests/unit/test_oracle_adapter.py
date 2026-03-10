"""
Phase 2 — Ingestion
OracleAdapter tests written BEFORE implementation (TDD).
"""

import pytest
from pathlib import Path

from tailwhip.ingestion.oracle_adapter import OracleAdapter


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_column(columns, column_name):
    """Return the first column with the given name."""
    return next((c for c in columns if c.column_name == column_name), None)


def get_table_columns(columns, table_name):
    """Return all columns belonging to a table."""
    return [c for c in columns if c.table_name == table_name]


# ---------------------------------------------------------------------------
# Parsing de constraints
# ---------------------------------------------------------------------------

class TestOracleAdapterParseiaPk:

    def setup_method(self):
        self.adapter = OracleAdapter()
        self.columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")

    def test_coluna_pk_tem_is_pk_true(self):
        col = get_column(self.columns, "ID_USUARIO")
        assert col is not None
        assert col.is_pk is True

    def test_coluna_pk_tem_is_fk_false(self):
        col = get_column(self.columns, "ID_USUARIO")
        assert col.is_fk is False

    def test_coluna_pk_nao_tem_fk_ref(self):
        col = get_column(self.columns, "ID_USUARIO")
        assert col.fk_ref_table is None
        assert col.fk_ref_schema is None

    def test_multiplas_colunas_pk(self):
        """Tables with composite PK must have all columns marked."""
        cols = get_table_columns(self.columns, "APP_USUARIO")
        pks = [c for c in cols if c.is_pk]
        assert len(pks) >= 1


class TestOracleAdapterParseFkComOwner:

    def setup_method(self):
        self.adapter = OracleAdapter()
        self.columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")

    def test_coluna_fk_tem_is_fk_true(self):
        cols = get_table_columns(self.columns, "APP_PEDIDO")
        fk_col = next((c for c in cols if c.is_fk), None)
        assert fk_col is not None
        assert fk_col.is_fk is True

    def test_coluna_fk_tem_is_pk_false(self):
        cols = get_table_columns(self.columns, "APP_PEDIDO")
        fk_col = next((c for c in cols if c.is_fk), None)
        assert fk_col.is_pk is False

    def test_coluna_fk_extrai_tabela_referenciada(self):
        cols = get_table_columns(self.columns, "APP_PEDIDO")
        fk_col = next((c for c in cols if c.is_fk), None)
        assert fk_col.fk_ref_table == "APP_USUARIO"

    def test_coluna_fk_extrai_schema_referenciado(self):
        cols = get_table_columns(self.columns, "APP_PEDIDO")
        fk_col = next((c for c in cols if c.is_fk), None)
        assert fk_col.fk_ref_schema == "APPOWNER"

    def test_formato_fk_com_owner_diferente(self):
        """FK pointing to another owner must be extracted correctly."""
        cols = get_table_columns(self.columns, "APP_USUARIO")
        fk_col = next((c for c in cols if c.is_fk), None)
        assert fk_col is not None
        assert fk_col.fk_ref_schema is not None
        assert fk_col.fk_ref_table is not None


class TestOracleAdapterColunaSemConstraint:

    def setup_method(self):
        self.adapter = OracleAdapter()
        self.columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")

    def test_coluna_normal_tem_is_pk_false(self):
        col = get_column(self.columns, "NOME")
        assert col is not None
        assert col.is_pk is False

    def test_coluna_normal_tem_is_fk_false(self):
        col = get_column(self.columns, "NOME")
        assert col.is_fk is False

    def test_coluna_normal_sem_fk_ref(self):
        col = get_column(self.columns, "NOME")
        assert col.fk_ref_table is None
        assert col.fk_ref_schema is None

    def test_constraint_vazio_nao_quebra(self):
        """Rows with empty CONSTRAINT_TYPE must be processed normally."""
        col = get_column(self.columns, "EMAIL")
        assert col is not None


# ---------------------------------------------------------------------------
# Encoding e formato do arquivo
# ---------------------------------------------------------------------------

class TestOracleAdapterEncodingLatin1:

    def setup_method(self):
        self.adapter = OracleAdapter()

    def test_le_arquivo_com_encoding_latin1(self):
        """Must read the CSV without UnicodeDecodeError."""
        columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")
        assert len(columns) > 0

    def test_separador_ponto_e_virgula(self):
        """Must parse correctly with sep=';'."""
        columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")
        assert any(c.table_name == "APP_USUARIO" for c in columns)

    def test_retorna_lista_de_column_metadata(self):
        """Must return a list of ColumnMetadata."""
        from tailwhip.models.column_metadata import ColumnMetadata
        columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")
        assert all(isinstance(c, ColumnMetadata) for c in columns)

    def test_arquivo_nao_encontrado_levanta_erro(self):
        """Non-existent file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            self.adapter.parse(Path("nao_existe.csv"))

    def test_owner_mapeado_para_schema_name(self):
        """OWNER field must be mapped to schema_name."""
        columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")
        col = get_column(columns, "ID_USUARIO")
        assert col.schema_name == "APPOWNER"

    def test_table_comment_mapeado_para_description(self):
        """TABLE_COMMENT must appear in the columns of the corresponding table."""
        columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")
        cols_usuario = get_table_columns(columns, "APP_USUARIO")
        assert any(c.description is not None for c in cols_usuario)

    def test_todas_as_tabelas_do_fixture_presentes(self):
        """All tables in oracle_sample.csv must be read."""
        columns = self.adapter.parse(FIXTURES_DIR / "oracle_sample.csv")
        tabelas = {c.table_name for c in columns}
        assert "APP_USUARIO" in tabelas
        assert "APP_PEDIDO" in tabelas
        assert "APP_DADOS_USUARIO" in tabelas
        assert "APP_STATUS" in tabelas


class TestParseConstraintFkSemOwner:
    """Branch: FK reference without schema (no owner prefix)."""

    def setup_method(self):
        self.adapter = OracleAdapter()

    def test_fk_sem_owner_preenche_fk_ref_table(self):
        """FK -> TABLE_NAME (no schema) must populate fk_ref_table and leave fk_ref_schema as None."""
        is_pk, is_fk, fk_ref_schema, fk_ref_table = self.adapter._parse_constraint(
            "FK -> APP_TABELA"
        )
        assert is_fk is True
        assert fk_ref_table == "APP_TABELA"

    def test_fk_sem_owner_deixa_schema_none(self):
        """FK -> TABLE_NAME without schema must result in fk_ref_schema=None."""
        _, _, fk_ref_schema, _ = self.adapter._parse_constraint("FK -> APP_TABELA")
        assert fk_ref_schema is None

    def test_fk_sem_owner_nao_eh_pk(self):
        """FK without owner must not activate is_pk."""
        is_pk, _, _, _ = self.adapter._parse_constraint("FK -> APP_TABELA")
        assert is_pk is False
