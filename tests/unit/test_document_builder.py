"""
Phase 3 — DocumentBuilder
Tests written BEFORE implementation (TDD).
"""

import pytest

from tailwhip.models.table_metadata import TableMetadata
from tailwhip.models.column_metadata import ColumnMetadata
from tailwhip.models.schema_document import SchemaDocument
from tailwhip.indexing.document_builder import DocumentBuilder


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture
def builder():
    return DocumentBuilder()


@pytest.fixture
def table_com_fk_in():
    return TableMetadata(
        table_name="APP_USUARIO",
        schema_name="APPOWNER",
        description="User registry",
        column_count=3,
        fk_in_count=5,
    )


@pytest.fixture
def table_sem_comentario():
    return TableMetadata(
        table_name="APP_DADOS_USUARIO",
        schema_name="APPOWNER",
        description=None,
        column_count=2,
        fk_in_count=0,
    )


@pytest.fixture
def colunas_completas(table_com_fk_in):
    return [
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="ID_USUARIO",
            data_type="NUMBER(10)",
            description="Unique identifier",
            is_pk=True,
        ),
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="NOME",
            data_type="VARCHAR2(100)",
            description="Full name",
        ),
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="ID_PERFIL",
            data_type="NUMBER(10)",
            is_fk=True,
            fk_ref_schema="APPOWNER",
            fk_ref_table="APP_PERFIL",
        ),
    ]


@pytest.fixture
def colunas_sem_comentario():
    return [
        ColumnMetadata(
            table_name="APP_DADOS_USUARIO",
            schema_name="APPOWNER",
            column_name="DATA_NASCIMENTO",
            data_type="DATE",
        ),
        ColumnMetadata(
            table_name="APP_DADOS_USUARIO",
            schema_name="APPOWNER",
            column_name="CPF",
            data_type="VARCHAR2(11)",
        ),
    ]


# ---------------------------------------------------------------------------
# test_documento_sem_comentario_nao_vazio
# ---------------------------------------------------------------------------

class TestDocumentoSemComentarioNaoVazio:

    def test_build_retorna_schema_document(self, builder, table_sem_comentario, colunas_sem_comentario):
        """build() must return a SchemaDocument instance."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        assert isinstance(doc, SchemaDocument)

    def test_doc_id_formato_correto(self, builder, table_sem_comentario, colunas_sem_comentario):
        """doc_id must follow the 'SCHEMA.TABLE_NAME' format."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        assert doc.doc_id == "APPOWNER.APP_DADOS_USUARIO"

    def test_doc_id_sem_schema(self, builder, colunas_sem_comentario):
        """Table without schema must generate doc_id with table_name only."""
        table = TableMetadata(table_name="APP_DADOS_USUARIO", schema_name=None)
        doc = builder.build(table, colunas_sem_comentario)
        assert doc.doc_id == "APP_DADOS_USUARIO"

    def test_embedding_text_nao_vazio_sem_comentarios(self, builder, table_sem_comentario, colunas_sem_comentario):
        """Table with no comments must generate non-empty embedding text."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        assert doc.to_embedding_text().strip() != ""

    def test_embedding_text_contem_nome_tokenizado(self, builder, table_sem_comentario, colunas_sem_comentario):
        """Table name must appear tokenized in the embedding text."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        assert "app dados usuario" in doc.to_embedding_text().lower()

    def test_embedding_text_contem_colunas_tokenizadas(self, builder, table_sem_comentario, colunas_sem_comentario):
        """Column names must appear tokenized."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        texto = doc.to_embedding_text().lower()
        assert "data nascimento" in texto
        assert "cpf" in texto

    def test_embedding_text_contem_data_types(self, builder, table_sem_comentario, colunas_sem_comentario):
        """Data types must be present in the embedding text."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        texto = doc.to_embedding_text().lower()
        assert "date" in texto

    def test_colunas_preservadas_no_documento(self, builder, table_sem_comentario, colunas_sem_comentario):
        """Columns must be accessible in the document."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        assert len(doc.columns) == 2


# ---------------------------------------------------------------------------
# test_documento_inclui_referenced_by
# ---------------------------------------------------------------------------

class TestDocumentoIncluiReferencedBy:

    def test_referenced_by_presente_quando_fk_in_count_maior_zero(
        self, builder, table_com_fk_in, colunas_completas
    ):
        """'Referenced by N tables' must appear when fk_in_count > 0."""
        doc = builder.build(table_com_fk_in, colunas_completas)
        assert "referenced by" in doc.to_embedding_text().lower()

    def test_referenced_by_contem_contagem_correta(
        self, builder, table_com_fk_in, colunas_completas
    ):
        """The count must match the table's fk_in_count."""
        doc = builder.build(table_com_fk_in, colunas_completas)
        assert "5" in doc.to_embedding_text()

    def test_referenced_by_ausente_quando_fk_in_count_zero(
        self, builder, table_sem_comentario, colunas_sem_comentario
    ):
        """Must not mention references when fk_in_count = 0."""
        doc = builder.build(table_sem_comentario, colunas_sem_comentario)
        assert "referenced by" not in doc.to_embedding_text().lower()

    def test_build_em_lote_retorna_lista_de_documentos(
        self, builder, table_com_fk_in, table_sem_comentario,
        colunas_completas, colunas_sem_comentario
    ):
        """build_all() must accept multiple tables and return a list of SchemaDocument."""
        pares = [
            (table_com_fk_in, colunas_completas),
            (table_sem_comentario, colunas_sem_comentario),
        ]
        docs = builder.build_all(pares)
        assert len(docs) == 2
        assert all(isinstance(d, SchemaDocument) for d in docs)

    def test_build_all_lista_vazia(self, builder):
        """build_all() with an empty list must return an empty list."""
        assert builder.build_all([]) == []

    def test_doc_ids_unicos_no_lote(
        self, builder, table_com_fk_in, table_sem_comentario,
        colunas_completas, colunas_sem_comentario
    ):
        """Each document in the batch must have a unique doc_id."""
        pares = [
            (table_com_fk_in, colunas_completas),
            (table_sem_comentario, colunas_sem_comentario),
        ]
        docs = builder.build_all(pares)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))