"""
Phase 2 — Ingestion
SchemaNormalizer tests written BEFORE implementation (TDD).
"""

import pytest

from tailwhip.models.column_metadata import ColumnMetadata
from tailwhip.ingestion.schema_normalizer import SchemaNormalizer


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture
def colunas_duas_tabelas():
    """Three tables: APP_USUARIO (referenced), APP_PEDIDO (has FK), APP_STATUS (lookup)."""
    return [
        # APP_USUARIO
        ColumnMetadata(table_name="APP_USUARIO", schema_name="OWN", column_name="ID_USUARIO", data_type="NUMBER", is_pk=True),
        ColumnMetadata(table_name="APP_USUARIO", schema_name="OWN", column_name="NOME", data_type="VARCHAR2(100)"),
        ColumnMetadata(table_name="APP_USUARIO", schema_name="OWN", column_name="EMAIL", data_type="VARCHAR2(200)"),
        # APP_PEDIDO — has FK pointing to APP_USUARIO
        ColumnMetadata(table_name="APP_PEDIDO", schema_name="OWN", column_name="ID_PEDIDO", data_type="NUMBER", is_pk=True),
        ColumnMetadata(table_name="APP_PEDIDO", schema_name="OWN", column_name="ID_USUARIO", data_type="NUMBER", is_fk=True, fk_ref_schema="OWN", fk_ref_table="APP_USUARIO"),
        ColumnMetadata(table_name="APP_PEDIDO", schema_name="OWN", column_name="VALOR", data_type="NUMBER(15,2)"),
        # APP_LOG — also has FK pointing to APP_USUARIO
        ColumnMetadata(table_name="APP_LOG", schema_name="OWN", column_name="ID_LOG", data_type="NUMBER", is_pk=True),
        ColumnMetadata(table_name="APP_LOG", schema_name="OWN", column_name="ID_USUARIO", data_type="NUMBER", is_fk=True, fk_ref_schema="OWN", fk_ref_table="APP_USUARIO"),
        # APP_STATUS — no FKs (lookup table)
        ColumnMetadata(table_name="APP_STATUS", schema_name="OWN", column_name="COD", data_type="VARCHAR2(5)", is_pk=True),
        ColumnMetadata(table_name="APP_STATUS", schema_name="OWN", column_name="DESCRICAO", data_type="VARCHAR2(50)"),
    ]


@pytest.fixture
def normalizer():
    return SchemaNormalizer()


# ---------------------------------------------------------------------------
# Agrupamento por tabela
# ---------------------------------------------------------------------------

class TestSchemaNormalizerAgrupaColumnasPorTabela:

    def test_retorna_lista_de_table_metadata(self, normalizer, colunas_duas_tabelas):
        from tailwhip.models.table_metadata import TableMetadata
        tables = normalizer.normalize(colunas_duas_tabelas)
        assert all(isinstance(t, TableMetadata) for t in tables)

    def test_numero_correto_de_tabelas(self, normalizer, colunas_duas_tabelas):
        tables = normalizer.normalize(colunas_duas_tabelas)
        assert len(tables) == 4  # APP_USUARIO, APP_PEDIDO, APP_LOG, APP_STATUS

    def test_table_names_corretos(self, normalizer, colunas_duas_tabelas):
        tables = normalizer.normalize(colunas_duas_tabelas)
        names = {t.table_name for t in tables}
        assert "APP_USUARIO" in names
        assert "APP_PEDIDO" in names
        assert "APP_LOG" in names
        assert "APP_STATUS" in names

    def test_column_count_correto(self, normalizer, colunas_duas_tabelas):
        tables = normalizer.normalize(colunas_duas_tabelas)
        usuario = next(t for t in tables if t.table_name == "APP_USUARIO")
        assert usuario.column_count == 3

    def test_schema_name_preservado(self, normalizer, colunas_duas_tabelas):
        tables = normalizer.normalize(colunas_duas_tabelas)
        usuario = next(t for t in tables if t.table_name == "APP_USUARIO")
        assert usuario.schema_name == "OWN"

    def test_lista_vazia_retorna_vazio(self, normalizer):
        tables = normalizer.normalize([])
        assert tables == []


# ---------------------------------------------------------------------------
# Cálculo de fk_in_count
# ---------------------------------------------------------------------------

class TestSchemaNormalizerCalculaFkInCount:

    def test_tabela_referenciada_tem_fk_in_count_correto(self, normalizer, colunas_duas_tabelas):
        """APP_USUARIO is referenced by APP_PEDIDO and APP_LOG — fk_in_count = 2."""
        tables = normalizer.normalize(colunas_duas_tabelas)
        usuario = next(t for t in tables if t.table_name == "APP_USUARIO")
        assert usuario.fk_in_count == 2

    def test_tabela_sem_referencias_tem_fk_in_count_zero(self, normalizer, colunas_duas_tabelas):
        """APP_STATUS is not referenced by anyone."""
        tables = normalizer.normalize(colunas_duas_tabelas)
        status = next(t for t in tables if t.table_name == "APP_STATUS")
        assert status.fk_in_count == 0

    def test_tabela_que_referencia_outra_tem_fk_in_count_zero(self, normalizer, colunas_duas_tabelas):
        """APP_PEDIDO points FK to APP_USUARIO but nobody points to APP_PEDIDO."""
        tables = normalizer.normalize(colunas_duas_tabelas)
        pedido = next(t for t in tables if t.table_name == "APP_PEDIDO")
        assert pedido.fk_in_count == 0

    def test_fk_in_count_nao_conta_a_propria_tabela(self, normalizer):
        """Self-references must not increment fk_in_count."""
        colunas = [
            ColumnMetadata(table_name="APP_CATEGORIA", schema_name="OWN", column_name="ID", data_type="NUMBER", is_pk=True),
            ColumnMetadata(table_name="APP_CATEGORIA", schema_name="OWN", column_name="ID_PAI", data_type="NUMBER", is_fk=True, fk_ref_schema="OWN", fk_ref_table="APP_CATEGORIA"),
        ]
        tables = normalizer.normalize(colunas)
        categoria = next(t for t in tables if t.table_name == "APP_CATEGORIA")
        assert categoria.fk_in_count == 0

    def test_fk_ref_para_tabela_fora_do_schema_ignorada(self, normalizer):
        """FK pointing to a table not in the CSV must not break."""
        colunas = [
            ColumnMetadata(table_name="APP_PEDIDO", schema_name="OWN", column_name="ID", data_type="NUMBER", is_pk=True),
            ColumnMetadata(table_name="APP_PEDIDO", schema_name="OWN", column_name="ID_EXT", data_type="NUMBER", is_fk=True, fk_ref_schema="OTHER", fk_ref_table="EXTERNAL_TABLE"),
        ]
        tables = normalizer.normalize(colunas)
        assert len(tables) == 1
        assert tables[0].fk_in_count == 0
