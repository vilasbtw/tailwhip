"""
Phase 6 — Integration
Tests the full tailwhip pipeline using the fictional oracle_sample.csv fixture.

These tests instantiate real components (ChromaDB, BM25, sentence-transformers).
They are slower than unit tests — run only in CI or explicitly.

To run unit tests only:
    pytest tests/unit/

To run everything including integration:
    pytest
"""

import pytest
import tempfile
from pathlib import Path

from tailwhip.ingestion.oracle_adapter import OracleAdapter
from tailwhip.ingestion.schema_normalizer import SchemaNormalizer
from tailwhip.indexing.document_builder import DocumentBuilder
from tailwhip.indexing.index_manager import IndexManager
from tailwhip.search.orchestrator import SearchOrchestrator
from tailwhip.models.search_result import SearchResult


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """
    Builds the full pipeline once for all tests in the module.
    Uses a temporary directory for ChromaDB and BM25 — does not pollute ~/.tailwhip.
    """
    tmp_dir = tmp_path_factory.mktemp("tailwhip_test")

    # 1. Ingestion
    adapter    = OracleAdapter()
    normalizer = SchemaNormalizer()
    builder    = DocumentBuilder()

    columns = adapter.parse(FIXTURES_DIR / "oracle_sample.csv")
    tables  = normalizer.normalize(columns)

    # Group columns by table for the builder
    from collections import defaultdict
    cols_by_table: dict[str, list] = defaultdict(list)
    for col in columns:
        cols_by_table[col.table_name].append(col)

    pairs     = [(table, cols_by_table[table.table_name]) for table in tables]
    documents = builder.build_all(pairs)

    # 2. Indexing
    index_manager = IndexManager(base_dir=tmp_dir)
    index_manager.build(documents)

    # 3. Orchestrator
    orchestrator = SearchOrchestrator(index_manager=index_manager)

    return orchestrator


# ---------------------------------------------------------------------------
# test_pipeline_completo_csv_ficticio
# ---------------------------------------------------------------------------

class TestPipelineCompletoCSVFicticio:

    def test_pipeline_inicializa_sem_erro(self, pipeline):
        """Full pipeline must initialize without exceptions."""
        assert pipeline is not None

    def test_search_retorna_lista_de_search_result(self, pipeline):
        """Real search must return a list of SearchResult."""
        results = pipeline.search("usuario")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_retorna_resultados_nao_vazios(self, pipeline):
        """Search for a term present in the fixture must return results."""
        results = pipeline.search("usuario")
        assert len(results) > 0

    def test_search_top_n_respeitado(self, pipeline):
        """top_n must limit the results."""
        results = pipeline.search("usuario", top_n=2)
        assert len(results) <= 2

    def test_rank_sequencial_a_partir_de_1(self, pipeline):
        """Ranks must be 1, 2, 3... without gaps."""
        results = pipeline.search("usuario", top_n=5)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_final_score_entre_0_e_1(self, pipeline):
        """All scores must be within the valid model range."""
        results = pipeline.search("usuario", top_n=5)
        assert all(0.0 <= r.final_score <= 1.0 for r in results)

    def test_scores_ordenados_decrescente(self, pipeline):
        """Results must come from highest to lowest score."""
        results = pipeline.search("usuario", top_n=5)
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_app_usuario_aparece_na_busca_por_usuario(self, pipeline):
        """APP_USUARIO must appear in results when searching for 'usuario'."""
        results = pipeline.search("usuario", top_n=10)
        table_names = [r.table.table_name for r in results]
        assert "APP_USUARIO" in table_names

    def test_app_usuario_supera_app_dados_usuario(self, pipeline):
        """
        APP_USUARIO (fk_in_count=2) must rank above
        APP_DADOS_USUARIO (fk_in_count=0) when searching for 'usuario'.

        This is the core use case — FK boost resolving
        ambiguity between tables with similar names.
        """
        results = pipeline.search("usuario", top_n=10)
        table_names = [r.table.table_name for r in results]

        if "APP_USUARIO" in table_names and "APP_DADOS_USUARIO" in table_names:
            pos_usuario       = table_names.index("APP_USUARIO")
            pos_dados_usuario = table_names.index("APP_DADOS_USUARIO")
            assert pos_usuario < pos_dados_usuario

    def test_busca_por_pedido_retorna_app_pedido(self, pipeline):
        """Search for 'pedido' must find APP_PEDIDO."""
        results = pipeline.search("pedido", top_n=5)
        table_names = [r.table.table_name for r in results]
        assert "APP_PEDIDO" in table_names

    def test_filtra_por_owner(self, pipeline):
        """Owner filter must return only tables from the specified owner."""
        results = pipeline.search("usuario", owners=["APPOWNER"], top_n=10)
        assert all(r.table.schema_name == "APPOWNER" for r in results)

    def test_owner_inexistente_retorna_vazio(self, pipeline):
        """Owner not present in the fixture must return an empty list."""
        results = pipeline.search("qualquer", owners=["OWNER_FAKE"])
        assert results == []

    def test_resultado_tem_colunas(self, pipeline):
        """Each result must have at least one column."""
        results = pipeline.search("usuario", top_n=3)
        assert all(len(r.relevant_columns) > 0 for r in results)

    def test_doc_ids_unicos_no_resultado(self, pipeline):
        """There must be no duplicate tables in the results."""
        results = pipeline.search("usuario", top_n=10)
        doc_ids = [r.doc_id for r in results]
        assert len(doc_ids) == len(set(doc_ids))

    def test_busca_em_portugues_funciona(self, pipeline):
        """Multilingual model must respond to PT-BR queries."""
        results = pipeline.search("cadastro de pessoas", top_n=5)
        assert len(results) > 0

    def test_segunda_busca_sem_reindexar(self, pipeline):
        """Pipeline must respond to multiple searches without re-indexing."""
        r1 = pipeline.search("usuario", top_n=3)
        r2 = pipeline.search("pedido", top_n=3)
        assert len(r1) > 0
        assert len(r2) > 0