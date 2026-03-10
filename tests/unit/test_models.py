"""
Phase 1 — Models
Tests written BEFORE implementation (TDD).
All should fail until the models are implemented.
"""

import pytest
from pydantic import ValidationError

from tailwhip.models.table_metadata import TableMetadata
from tailwhip.models.column_metadata import ColumnMetadata
from tailwhip.models.schema_document import SchemaDocument, tokenize_name
from tailwhip.models.search_result import SearchResult


# ---------------------------------------------------------------------------
# TableMetadata
# ---------------------------------------------------------------------------

class TestTableMetadata:

    def test_campos_obrigatorios(self):
        """Only table_name is required."""
        table = TableMetadata(table_name="APP_USUARIO")
        assert table.table_name == "APP_USUARIO"

    def test_campos_opcionais_tem_defaults(self):
        """Optional fields must have defaults that don't break the pipeline."""
        table = TableMetadata(table_name="APP_USUARIO")
        assert table.schema_name is None
        assert table.description is None
        assert table.column_count == 0
        assert table.fk_in_count == 0

    def test_todos_os_campos(self):
        """Accepts all fields populated."""
        table = TableMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            description="User registry",
            column_count=5,
            fk_in_count=32,
        )
        assert table.schema_name == "APPOWNER"
        assert table.fk_in_count == 32

    def test_table_name_nao_pode_ser_vazio(self):
        """Empty table_name must be rejected."""
        with pytest.raises(ValidationError):
            TableMetadata(table_name="")

    def test_fk_in_count_nao_pode_ser_negativo(self):
        """Negative centrality makes no sense."""
        with pytest.raises(ValidationError):
            TableMetadata(table_name="APP_USUARIO", fk_in_count=-1)

    def test_column_count_nao_pode_ser_negativo(self):
        with pytest.raises(ValidationError):
            TableMetadata(table_name="APP_USUARIO", column_count=-1)


# ---------------------------------------------------------------------------
# ColumnMetadata
# ---------------------------------------------------------------------------

class TestColumnMetadata:

    def test_campos_obrigatorios(self):
        col = ColumnMetadata(
            table_name="APP_USUARIO",
            column_name="NOME",
            data_type="VARCHAR2(100)",
        )
        assert col.column_name == "NOME"

    def test_campos_opcionais_tem_defaults(self):
        col = ColumnMetadata(
            table_name="APP_USUARIO",
            column_name="NOME",
            data_type="VARCHAR2(100)",
        )
        assert col.schema_name is None
        assert col.description is None
        assert col.is_pk is False
        assert col.is_fk is False
        assert col.fk_ref_schema is None
        assert col.fk_ref_table is None

    def test_coluna_pk(self):
        col = ColumnMetadata(
            table_name="APP_USUARIO",
            column_name="ID_USUARIO",
            data_type="NUMBER(10)",
            is_pk=True,
        )
        assert col.is_pk is True
        assert col.is_fk is False

    def test_coluna_fk_com_referencia(self):
        col = ColumnMetadata(
            table_name="APP_PEDIDO",
            column_name="ID_USUARIO",
            data_type="NUMBER(10)",
            is_fk=True,
            fk_ref_schema="APPOWNER",
            fk_ref_table="APP_USUARIO",
        )
        assert col.is_fk is True
        assert col.fk_ref_table == "APP_USUARIO"
        assert col.fk_ref_schema == "APPOWNER"

    def test_column_name_nao_pode_ser_vazio(self):
        with pytest.raises(ValidationError):
            ColumnMetadata(table_name="APP_USUARIO", column_name="", data_type="NUMBER")

    def test_data_type_nao_pode_ser_vazio(self):
        with pytest.raises(ValidationError):
            ColumnMetadata(table_name="APP_USUARIO", column_name="ID", data_type="")


# ---------------------------------------------------------------------------
# tokenize_name
# ---------------------------------------------------------------------------

class TestTokenizeName:

    def test_separa_por_underscore(self):
        assert tokenize_name("APP_USUARIO") == "app usuario"

    def test_multiplos_underscores(self):
        assert tokenize_name("APP_DADOS_USUARIO_EXT") == "app dados usuario ext"

    def test_converte_para_minusculo(self):
        assert tokenize_name("APP_USUARIO_ACESSO") == "app usuario acesso"

    def test_nome_simples_sem_underscore(self):
        assert tokenize_name("USUARIOS") == "usuarios"

    def test_nome_ja_minusculo(self):
        assert tokenize_name("app_usuario") == "app usuario"


# ---------------------------------------------------------------------------
# SchemaDocument — test_schema_document_tokeniza_nome
# ---------------------------------------------------------------------------

class TestSchemaDocumentTokenizaNome:

    def test_embedding_text_tokeniza_table_name(self, documento_com_comentario):
        """Table name must appear tokenized in the embedding text."""
        texto = documento_com_comentario.to_embedding_text()
        assert "app usuario" in texto.lower()
        assert "APP_USUARIO" not in texto

    def test_embedding_text_tokeniza_column_names(self, documento_com_comentario):
        """Column names must appear tokenized."""
        texto = documento_com_comentario.to_embedding_text()
        assert "id usuario" in texto.lower()
        assert "ID_USUARIO" not in texto

    def test_embedding_text_inclui_schema_name(self, documento_com_comentario):
        """Schema/owner must appear in the text."""
        texto = documento_com_comentario.to_embedding_text()
        assert "APPOWNER" in texto


# ---------------------------------------------------------------------------
# SchemaDocument — test_schema_document_texto_sem_comentarios
# ---------------------------------------------------------------------------

class TestSchemaDocumentTextoSemComentarios:

    def test_texto_nao_vazio_sem_nenhum_comentario(self, documento_sem_comentario):
        """Table with no comment must generate useful text via tokenization."""
        texto = documento_sem_comentario.to_embedding_text()
        assert texto.strip() != ""

    def test_texto_contem_nomes_tokenizados_das_colunas(self, documento_sem_comentario):
        """Even without comments, tokenized column names must be present."""
        texto = documento_sem_comentario.to_embedding_text()
        assert "data nascimento" in texto.lower()
        assert "cpf" in texto.lower()

    def test_texto_contem_tipos_de_dados(self, documento_sem_comentario):
        """Data types must be present to help with search."""
        texto = documento_sem_comentario.to_embedding_text()
        assert "DATE" in texto or "date" in texto.lower()

    def test_referenced_by_ausente_quando_fk_in_count_zero(self, documento_sem_comentario):
        """Must not mention references when fk_in_count is 0."""
        texto = documento_sem_comentario.to_embedding_text()
        assert "referenced by" not in texto.lower()

    def test_referenced_by_presente_quando_fk_in_count_maior_zero(self, documento_com_comentario):
        """Must mention the number of references when fk_in_count > 0."""
        texto = documento_com_comentario.to_embedding_text()
        assert "referenced by" in texto.lower()
        assert "2" in texto  # fk_in_count=2 in the fixture


# ---------------------------------------------------------------------------
# SchemaDocument — test_schema_document_destaca_pk_fk
# ---------------------------------------------------------------------------

class TestSchemaDocumentDestacaPkFk:

    def test_coluna_pk_destacada_no_texto(self, documento_com_comentario):
        """PK columns must be identified in the embedding text."""
        texto = documento_com_comentario.to_embedding_text()
        assert "[PK]" in texto or "pk" in texto.lower()

    def test_coluna_fk_destacada_no_texto(self, documento_com_comentario):
        """FK columns must be identified in the embedding text."""
        texto = documento_com_comentario.to_embedding_text()
        assert "[FK]" in texto or "fk" in texto.lower()

    def test_fk_referencia_tabela_destino_no_texto(self, documento_com_comentario):
        """The table referenced by the FK must appear in the text."""
        texto = documento_com_comentario.to_embedding_text()
        assert "app perfil" in texto.lower() or "APP_PERFIL" in texto

    def test_colunas_sem_constraint_nao_tem_marcador(self, documento_com_comentario):
        """Regular columns must not have PK or FK markers."""
        texto = documento_com_comentario.to_embedding_text()
        assert texto.count("[PK]") == 1
        assert texto.count("[FK ->") == 1


# ---------------------------------------------------------------------------
# SchemaDocument — doc_id
# ---------------------------------------------------------------------------

class TestSchemaDocumentDocId:

    def test_doc_id_formato_schema_ponto_tabela(self, documento_com_comentario):
        assert documento_com_comentario.doc_id == "APPOWNER.APP_USUARIO"

    def test_doc_id_nao_pode_ser_vazio(self):
        with pytest.raises(ValidationError):
            SchemaDocument(
                doc_id="",
                table=TableMetadata(table_name="APP_USUARIO"),
                columns=[],
            )


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:

    def test_campos_obrigatorios(self, table_com_comentario, colunas_app_usuario):
        result = SearchResult(
            rank=1,
            doc_id="APPOWNER.APP_USUARIO",
            table=table_com_comentario,
            relevant_columns=colunas_app_usuario,
            final_score=0.847,
        )
        assert result.rank == 1
        assert result.final_score == 0.847

    def test_rank_comeca_em_1(self, table_com_comentario):
        """Rank cannot be 0 or negative."""
        with pytest.raises(ValidationError):
            SearchResult(
                rank=0,
                doc_id="APPOWNER.APP_USUARIO",
                table=table_com_comentario,
                relevant_columns=[],
                final_score=0.5,
            )

    def test_final_score_entre_0_e_1(self, table_com_comentario):
        """Score outside [0, 1] must be rejected."""
        with pytest.raises(ValidationError):
            SearchResult(
                rank=1,
                doc_id="APPOWNER.APP_USUARIO",
                table=table_com_comentario,
                relevant_columns=[],
                final_score=1.5,
            )