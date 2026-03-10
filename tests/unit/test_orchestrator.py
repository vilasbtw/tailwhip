"""
Phase 5 — Orchestrator (with mocks)
Tests written BEFORE implementation (TDD).

The Orchestrator depends on IndexManager (ChromaDB + BM25 + embeddings).
We use mocks to isolate the orchestration logic without needing real indexes —
fast tests with no external dependencies.
"""

import pytest
from unittest.mock import MagicMock, patch

from tailwhip.models.table_metadata import TableMetadata
from tailwhip.models.column_metadata import ColumnMetadata
from tailwhip.models.schema_document import SchemaDocument
from tailwhip.models.search_result import SearchResult
from tailwhip.search.orchestrator import SearchOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_doc(table_name: str, schema: str, fk_in_count: int = 0) -> SchemaDocument:
    table = TableMetadata(
        table_name=table_name,
        schema_name=schema,
        fk_in_count=fk_in_count,
    )
    cols = [
        ColumnMetadata(
            table_name=table_name,
            schema_name=schema,
            column_name="ID",
            data_type="NUMBER",
            is_pk=True,
        )
    ]
    return SchemaDocument(
        doc_id=f"{schema}.{table_name}",
        table=table,
        columns=cols,
    )


@pytest.fixture
def docs():
    return [
        make_doc("APP_USUARIO",       "SCHEMA_A", fk_in_count=32),
        make_doc("APP_DADOS_USUARIO", "SCHEMA_A", fk_in_count=1),
        make_doc("APP_PARCEIRO",      "SCHEMA_A", fk_in_count=104),
        make_doc("APP_NOTA_FISCAL",   "SCHEMA_B", fk_in_count=5),
        make_doc("APP_ITEM_NOTA",     "SCHEMA_B", fk_in_count=2),
    ]


@pytest.fixture
def mock_index_manager(docs):
    """
    Mock IndexManager that returns pre-defined rankings.
    vector_search and bm25_search return doc_ids in order.
    get_documents returns the full document dict.
    """
    manager = MagicMock()

    all_doc_ids = [d.doc_id for d in docs]
    doc_map = {d.doc_id: d for d in docs}

    manager.vector_search.return_value = all_doc_ids
    manager.bm25_search.return_value = all_doc_ids
    manager.get_documents.return_value = doc_map

    return manager


@pytest.fixture
def dummy_expander():
    """Expander that returns the query unchanged — isolates tests from synonyms.json."""
    from tailwhip.search.query_expander import QueryExpander
    expander = MagicMock(spec=QueryExpander)
    expander.expand.side_effect = lambda q: q
    return expander


@pytest.fixture
def orchestrator(mock_index_manager, dummy_expander):
    return SearchOrchestrator(index_manager=mock_index_manager, expander=dummy_expander)


# ---------------------------------------------------------------------------
# test_orchestrator_retorna_top_n
# ---------------------------------------------------------------------------

class TestOrchestratorRetornaTopN:

    def test_retorna_lista_de_search_result(self, orchestrator):
        results = orchestrator.search("usuario", top_n=3)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_retorna_exatamente_top_n(self, orchestrator):
        results = orchestrator.search("usuario", top_n=3)
        assert len(results) == 3

    def test_top_n_maior_que_disponiveis_retorna_todos(self, orchestrator, docs):
        results = orchestrator.search("usuario", top_n=100)
        assert len(results) == len(docs)

    def test_rank_comeca_em_1(self, orchestrator):
        results = orchestrator.search("usuario", top_n=3)
        assert results[0].rank == 1

    def test_rank_e_sequencial(self, orchestrator):
        results = orchestrator.search("usuario", top_n=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_final_score_entre_0_e_1(self, orchestrator):
        results = orchestrator.search("usuario", top_n=5)
        assert all(0.0 <= r.final_score <= 1.0 for r in results)

    def test_chama_vector_search_com_query(self, orchestrator, mock_index_manager):
        orchestrator.search("usuario", top_n=3)
        mock_index_manager.vector_search.assert_called_once_with("usuario")

    def test_chama_bm25_search_com_query(self, orchestrator, mock_index_manager):
        orchestrator.search("usuario", top_n=3)
        mock_index_manager.bm25_search.assert_called_once_with("usuario")

    def test_resultado_tem_doc_id_valido(self, orchestrator, docs):
        valid_ids = {d.doc_id for d in docs}
        results = orchestrator.search("usuario", top_n=5)
        assert all(r.doc_id in valid_ids for r in results)

    def test_resultado_tem_tabela_preenchida(self, orchestrator):
        results = orchestrator.search("usuario", top_n=3)
        assert all(r.table is not None for r in results)

    def test_top_n_default_e_10(self, orchestrator):
        """Default top_n must be 10."""
        results = orchestrator.search("usuario")
        assert len(results) <= 10


# ---------------------------------------------------------------------------
# test_orchestrator_filtra_por_owner
# ---------------------------------------------------------------------------

class TestOrchestratorFiltraPorOwner:

    def test_filtra_por_owner_unico(self, orchestrator):
        results = orchestrator.search("nota fiscal", owners=["SCHEMA_B"])
        assert all(r.table.schema_name == "SCHEMA_B" for r in results)

    def test_filtra_por_multiplos_owners(self, orchestrator):
        results = orchestrator.search("parceiro", owners=["SCHEMA_A", "SCHEMA_B"])
        assert all(r.table.schema_name in {"SCHEMA_A", "SCHEMA_B"} for r in results)

    def test_sem_filtro_retorna_todos_os_owners(self, orchestrator, docs):
        results = orchestrator.search("usuario", top_n=10)
        schemas = {r.table.schema_name for r in results}
        assert "SCHEMA_A" in schemas
        assert "SCHEMA_B" in schemas

    def test_owner_inexistente_retorna_lista_vazia(self, orchestrator):
        results = orchestrator.search("qualquer", owners=["OWNER_INEXISTENTE"])
        assert results == []

    def test_top_n_aplicado_apos_filtro_de_owner(self, orchestrator):
        """top_n must limit results AFTER the owner filter."""
        results = orchestrator.search("usuario", owners=["SCHEMA_A"], top_n=2)
        assert len(results) <= 2
        assert all(r.table.schema_name == "SCHEMA_A" for r in results)


# ---------------------------------------------------------------------------
# Phase 7 — PK boost in the orchestrator pipeline
# ---------------------------------------------------------------------------

class TestOrchestratorPkBoost:

    def test_tabela_pai_supera_tabela_filha_com_pk_boost(self, mock_index_manager):
        """
        FATURAS (ID_INVOICE as PK, fk_in_count=5) vs
        ITENS (ID_INVOICE as FK, fk_in_count=20).

        Without PK boost: ITENS wins due to higher FK count.
        With PK boost: FATURAS must rise because ID_INVOICE is its PK.
        """
        doc_faturas = SchemaDocument(
            doc_id="OWN.FATURAS",
            table=TableMetadata(table_name="FATURAS", schema_name="OWN", fk_in_count=5),
            columns=[
                ColumnMetadata(table_name="FATURAS", column_name="ID_INVOICE",  data_type="NUMBER", is_pk=True),
                ColumnMetadata(table_name="FATURAS", column_name="NUM_INVOICE", data_type="VARCHAR2(20)", is_pk=True),
            ],
        )
        doc_itens = SchemaDocument(
            doc_id="OWN.ITENS",
            table=TableMetadata(table_name="ITENS", schema_name="OWN", fk_in_count=20),
            columns=[
                ColumnMetadata(table_name="ITENS", column_name="NUM_ITEM",   data_type="NUMBER", is_pk=True),
                ColumnMetadata(table_name="ITENS", column_name="ID_INVOICE", data_type="NUMBER", is_fk=True),
            ],
        )

        doc_map = {"OWN.FATURAS": doc_faturas, "OWN.ITENS": doc_itens}
        ranking = ["OWN.FATURAS", "OWN.ITENS"]

        mock_index_manager.vector_search.return_value = ranking
        mock_index_manager.bm25_search.return_value   = ranking
        mock_index_manager.get_documents.return_value = doc_map

        orchestrator = SearchOrchestrator(index_manager=mock_index_manager)
        results = orchestrator.search("id invoice numero", top_n=2)

        table_names = [r.table.table_name for r in results]
        assert table_names[0] == "FATURAS"


class TestOrchestratorBranchesDefensivos:
    """Defensive branches of the orchestrator."""

    @pytest.fixture
    def dummy_expander(self):
        from tailwhip.search.query_expander import QueryExpander
        expander = MagicMock(spec=QueryExpander)
        expander.expand.side_effect = lambda q: q
        return expander

    def test_rrf_vazio_retorna_lista_vazia(self, dummy_expander):
        """When both indexes return empty, search must return []."""
        manager = MagicMock()
        manager.vector_search.return_value = []
        manager.bm25_search.return_value   = []

        orchestrator = SearchOrchestrator(index_manager=manager, expander=dummy_expander)
        results = orchestrator.search("anything")
        assert results == []

    def test_doc_id_ausente_no_doc_map_e_ignorado(self, dummy_expander):
        """doc_id present in ranking but missing from doc_map must not appear in results."""
        doc = SchemaDocument(
            doc_id="OWN.REAL",
            table=TableMetadata(table_name="REAL", schema_name="OWN", fk_in_count=1),
            columns=[ColumnMetadata(table_name="REAL", column_name="ID", data_type="NUMBER", is_pk=True)],
        )
        doc_map = {"OWN.REAL": doc}  # OWN.GHOST intentionally absent

        manager = MagicMock()
        manager.vector_search.return_value = ["OWN.REAL", "OWN.GHOST"]
        manager.bm25_search.return_value   = ["OWN.REAL", "OWN.GHOST"]
        manager.get_documents.return_value = doc_map

        orchestrator = SearchOrchestrator(index_manager=manager, expander=dummy_expander)
        results = orchestrator.search("real")

        doc_ids = [r.doc_id for r in results]
        assert "OWN.GHOST" not in doc_ids
        assert "OWN.REAL" in doc_ids