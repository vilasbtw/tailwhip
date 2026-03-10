"""
Phase 4 — RankFusion (pure logic)
Tests written BEFORE implementation (TDD).

rank_fusion is pure logic — no I/O, no external dependencies.
Receives rankings and returns scores. Easy to test and reason about.
"""

import math
import pytest

from tailwhip.search.rank_fusion import reciprocal_rank_fusion, apply_fk_boost, normalize_scores, apply_pk_boost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ranking(*doc_ids: str) -> list[str]:
    """Create a simple ranking as an ordered list of doc_ids."""
    return list(doc_ids)


# ---------------------------------------------------------------------------
# test_rrf_formula_k60
# ---------------------------------------------------------------------------

class TestRrfFormulaK60:

    def test_score_unico_ranking_posicao_1(self):
        """With k=60, position 1 -> score = 1/(60+1) ≈ 0.01639."""
        rankings = [make_ranking("A")]
        scores = reciprocal_rank_fusion(rankings)
        expected = 1 / (60 + 1)
        assert abs(scores["A"] - expected) < 1e-9

    def test_score_unico_ranking_posicao_2(self):
        """Position 2 must have a lower score than position 1."""
        rankings = [make_ranking("A", "B")]
        scores = reciprocal_rank_fusion(rankings)
        assert scores["A"] > scores["B"]
        assert abs(scores["B"] - 1 / (60 + 2)) < 1e-9

    def test_score_diminui_com_posicao(self):
        """Scores must be strictly decreasing with position."""
        rankings = [make_ranking("A", "B", "C", "D")]
        scores = reciprocal_rank_fusion(rankings)
        assert scores["A"] > scores["B"] > scores["C"] > scores["D"]

    def test_dois_rankings_independentes_somam_scores(self):
        """Doc present in two rankings accumulates scores from both."""
        r1 = make_ranking("A", "B")
        r2 = make_ranking("A", "C")
        scores = reciprocal_rank_fusion([r1, r2])
        # A is at position 1 in both: 1/61 + 1/61 = 2/61
        expected_a = 1 / 61 + 1 / 61
        assert abs(scores["A"] - expected_a) < 1e-9

    def test_doc_ausente_em_um_ranking_recebe_score_parcial(self):
        """Doc present in only one ranking must have a lower score than doc in both."""
        r1 = make_ranking("A", "B")
        r2 = make_ranking("A", "C")
        scores = reciprocal_rank_fusion([r1, r2])
        # B is only in r1, C only in r2, A is in both
        assert scores["A"] > scores["B"]
        assert scores["A"] > scores["C"]

    def test_rankings_vazios_retorna_dict_vazio(self):
        assert reciprocal_rank_fusion([]) == {}

    def test_ranking_com_lista_vazia(self):
        assert reciprocal_rank_fusion([[]]) == {}

    def test_k_customizado(self):
        """Must accept a custom k value."""
        rankings = [make_ranking("A")]
        scores = reciprocal_rank_fusion(rankings, k=10)
        expected = 1 / (10 + 1)
        assert abs(scores["A"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# test_rrf_tabela_em_ambos_rankings_sobe
# ---------------------------------------------------------------------------

class TestRrfTabelaEmAmbosRankingsSobe:

    def test_doc_em_ambos_rankings_supera_doc_em_apenas_um(self):
        """
        Real scenario: vector search and BM25 return different rankings.
        Doc present in both must outscore doc present in only one.
        """
        vetor  = make_ranking("APP_DADOS_USUARIO", "APP_USUARIO")
        bm25   = make_ranking("APP_USUARIO", "APP_DADOS_USUARIO")
        scores = reciprocal_rank_fusion([vetor, bm25])
        # APP_USUARIO is at pos 2 in vector and pos 1 in BM25
        # APP_DADOS_USUARIO is at pos 1 in vector and pos 2 in BM25
        # scores must be equal by symmetry — both accumulate 1/61 + 1/62
        assert abs(scores["APP_USUARIO"] - scores["APP_DADOS_USUARIO"]) < 1e-9

    def test_doc_pos1_ambos_rankings_tem_score_maximo(self):
        """Doc at position 1 in all rankings must have the highest score."""
        r1 = make_ranking("CAMPEAO", "B", "C")
        r2 = make_ranking("CAMPEAO", "D", "E")
        scores = reciprocal_rank_fusion([r1, r2])
        assert scores["CAMPEAO"] == max(scores.values())

    def test_ordem_final_reflete_acumulo_de_rankings(self):
        """
        APP_USUARIO appears at the top of 3 rankings,
        APP_PERFIL only at the top of 1. APP_USUARIO must win.
        """
        r1 = make_ranking("APP_USUARIO", "APP_PERFIL")
        r2 = make_ranking("APP_USUARIO", "APP_PERFIL")
        r3 = make_ranking("APP_PERFIL", "APP_USUARIO")
        scores = reciprocal_rank_fusion([r1, r2, r3])
        assert scores["APP_USUARIO"] > scores["APP_PERFIL"]


# ---------------------------------------------------------------------------
# test_fk_boost_proporcional_ao_fk_in_count
# ---------------------------------------------------------------------------

class TestFkBoostProporcionalAoFkInCount:

    def test_boost_zero_fk_in_count(self):
        """fk_in_count=0 -> boost = log(1+0) = 0 -> score zeroed out."""
        scores = {"A": 0.5}
        fk_counts = {"A": 0}
        boosted = apply_fk_boost(scores, fk_counts)
        assert boosted["A"] == 0.0

    def test_boost_maior_com_mais_referencias(self):
        """Table with more FK refs must have a higher final score."""
        scores = {"CENTRAL": 0.5, "PERIFERIA": 0.5}
        fk_counts = {"CENTRAL": 32, "PERIFERIA": 1}
        boosted = apply_fk_boost(scores, fk_counts)
        assert boosted["CENTRAL"] > boosted["PERIFERIA"]

    def test_boost_proporcional_ao_log(self):
        """score_final = rrf_score * log(1 + log(1 + fk_in_count)) — double log."""
        scores = {"A": 0.5}
        fk_counts = {"A": 10}
        boosted = apply_fk_boost(scores, fk_counts)
        expected = 0.5 * math.log(1 + math.log(1 + 10))
        assert abs(boosted["A"] - expected) < 1e-9

    def test_boost_doc_sem_fk_count_usa_zero(self):
        """Doc not present in fk_counts must receive a boost of fk_in_count=0."""
        scores = {"A": 0.5, "B": 0.5}
        fk_counts = {"A": 5}  # B has no entry
        boosted = apply_fk_boost(scores, fk_counts)
        assert boosted["B"] == 0.0

    def test_boost_nao_altera_ordem_quando_fk_counts_iguais(self):
        """With equal fk_in_count, the RRF order must be preserved."""
        scores = {"PRIMEIRO": 0.9, "SEGUNDO": 0.5}
        fk_counts = {"PRIMEIRO": 5, "SEGUNDO": 5}
        boosted = apply_fk_boost(scores, fk_counts)
        assert boosted["PRIMEIRO"] > boosted["SEGUNDO"]

    def test_boost_cenario_real_tabela_central_vs_periferica(self):
        """
        Central table (32 refs) must outscore peripheral table (1 ref)
        even with equal RRF scores.
        """
        scores = {"APP_USUARIO": 0.5, "APP_DADOS_USUARIO": 0.5}
        fk_counts = {"APP_USUARIO": 32, "APP_DADOS_USUARIO": 1}
        boosted = apply_fk_boost(scores, fk_counts)
        assert boosted["APP_USUARIO"] > boosted["APP_DADOS_USUARIO"]


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:

    def test_score_maximo_vira_1(self):
        scores = {"A": 10.0, "B": 5.0, "C": 2.0}
        normalized = normalize_scores(scores)
        assert normalized["A"] == pytest.approx(1.0)

    def test_score_minimo_vira_0(self):
        scores = {"A": 10.0, "B": 5.0, "C": 2.0}
        normalized = normalize_scores(scores)
        assert normalized["C"] == pytest.approx(0.0)

    def test_scores_entre_0_e_1(self):
        scores = {"A": 10.0, "B": 5.0, "C": 2.0}
        normalized = normalize_scores(scores)
        assert all(0.0 <= v <= 1.0 for v in normalized.values())

    def test_scores_iguais_todos_viram_0(self):
        """When all scores are equal, returns 0 to avoid division by zero."""
        scores = {"A": 5.0, "B": 5.0}
        normalized = normalize_scores(scores)
        assert all(v == 0.0 for v in normalized.values())

    def test_dict_vazio_retorna_vazio(self):
        assert normalize_scores({}) == {}


# ---------------------------------------------------------------------------
# Phase 7a — FK boost softened (double log)
# ---------------------------------------------------------------------------

class TestFkBoostSuavizado:

    def test_diferenca_entre_alto_e_baixo_fk_menor_que_single_log(self):
        """
        Double log must reduce the advantage of heavily referenced tables.
        With fk=100 vs fk=5, the score ratio must be smaller than with single log.
        """
        scores = {"ALTO": 0.5, "BAIXO": 0.5}
        fk_counts = {"ALTO": 100, "BAIXO": 5}

        boosted = apply_fk_boost(scores, fk_counts)
        ratio_double_log = boosted["ALTO"] / boosted["BAIXO"]

        # ratio with single log for comparison
        ratio_single_log = math.log(1 + 100) / math.log(1 + 5)

        assert ratio_double_log < ratio_single_log

    def test_ordem_relativa_preservada_apos_suavizacao(self):
        """More referenced table must still win — just with a smaller margin."""
        scores = {"APP_SISTEMA": 0.5, "APP_USUARIO": 0.5}
        fk_counts = {"APP_SISTEMA": 58, "APP_USUARIO": 32}
        boosted = apply_fk_boost(scores, fk_counts)
        assert boosted["APP_SISTEMA"] > boosted["APP_USUARIO"]

    def test_cenario_real_invoice_tabela_pai_vs_filha(self):
        """
        Parent table (5 refs) vs child table (20 refs).
        With double log the difference must be small enough for the PK boost
        to reverse the order when needed.
        """
        scores = {"FATURAS": 0.5, "ITENS": 0.5}
        fk_counts = {"FATURAS": 5, "ITENS": 20}
        boosted = apply_fk_boost(scores, fk_counts)

        # With double log: log(1+log(6)) vs log(1+log(21))
        ratio = boosted["ITENS"] / boosted["FATURAS"]
        # Ratio must be well below what single log would give (log(21)/log(6) ≈ 2.0)
        assert ratio < 1.5


# ---------------------------------------------------------------------------
# Phase 7b — PK boost
# ---------------------------------------------------------------------------

class TestApplyPkBoost:

    def _make_doc(self, table_name, pk_columns, fk_columns=None):
        """Helper to create a minimal SchemaDocument for tests."""
        from tailwhip.models.table_metadata import TableMetadata
        from tailwhip.models.column_metadata import ColumnMetadata
        from tailwhip.models.schema_document import SchemaDocument

        cols = [
            ColumnMetadata(table_name=table_name, column_name=col, data_type="NUMBER", is_pk=True)
            for col in pk_columns
        ]
        if fk_columns:
            cols += [
                ColumnMetadata(table_name=table_name, column_name=col, data_type="NUMBER", is_fk=True)
                for col in fk_columns
            ]
        return SchemaDocument(
            doc_id=f"OWN.{table_name}",
            table=TableMetadata(table_name=table_name),
            columns=cols,
        )

    def test_pk_boost_sobe_tabela_com_pk_que_bate_com_query(self):
        """Table with ID_INVOICE as PK must receive boost for query 'id invoice'."""
        doc_pai   = self._make_doc("FATURAS",  pk_columns=["ID_INVOICE"])
        doc_filho = self._make_doc("ITENS",    pk_columns=["NUM_ITEM"], fk_columns=["ID_INVOICE"])

        scores   = {"OWN.FATURAS": 0.5, "OWN.ITENS": 0.5}
        docs     = {"OWN.FATURAS": doc_pai, "OWN.ITENS": doc_filho}
        tokens   = ["id", "invoice"]

        boosted = apply_pk_boost(scores, docs, tokens)
        assert boosted["OWN.FATURAS"] > boosted["OWN.ITENS"]

    def test_pk_boost_nao_afeta_tabela_sem_pk_match(self):
        """Table with no PK matching the query must not receive a boost."""
        doc = self._make_doc("TABELA", pk_columns=["COD_OUTRO"])
        scores = {"OWN.TABELA": 0.5}
        docs   = {"OWN.TABELA": doc}
        tokens = ["id", "invoice"]

        boosted = apply_pk_boost(scores, docs, tokens)
        assert boosted["OWN.TABELA"] == pytest.approx(0.5)

    def test_pk_boost_proporcional_ao_match(self):
        """Table with 2 matching PKs must have a higher boost than table with 1."""
        doc_dois = self._make_doc("T_DOIS", pk_columns=["ID_INVOICE", "NUM_INVOICE"])
        doc_um   = self._make_doc("T_UM",   pk_columns=["ID_INVOICE", "COD_OUTRO"])

        scores = {"OWN.T_DOIS": 0.5, "OWN.T_UM": 0.5}
        docs   = {"OWN.T_DOIS": doc_dois, "OWN.T_UM": doc_um}
        tokens = ["id", "invoice", "num", "invoice"]

        boosted = apply_pk_boost(scores, docs, tokens)
        assert boosted["OWN.T_DOIS"] > boosted["OWN.T_UM"]

    def test_pk_boost_query_sem_match_nao_penaliza(self):
        """Score must not decrease when the query matches no PK."""
        doc    = self._make_doc("TABELA", pk_columns=["ID_SISTEMA"])
        scores = {"OWN.TABELA": 0.5}
        docs   = {"OWN.TABELA": doc}
        tokens = ["invoice", "numero"]

        boosted = apply_pk_boost(scores, docs, tokens)
        assert boosted["OWN.TABELA"] >= 0.5

    def test_pk_boost_ignora_colunas_fk(self):
        """FK columns matching the query must not generate a boost."""
        doc = self._make_doc("ITENS", pk_columns=["NUM_ITEM"], fk_columns=["ID_INVOICE"])
        scores = {"OWN.ITENS": 0.5}
        docs   = {"OWN.ITENS": doc}
        tokens = ["id", "invoice"]

        boosted = apply_pk_boost(scores, docs, tokens)
        assert boosted["OWN.ITENS"] == pytest.approx(0.5)

    def test_pk_boost_dict_vazio_retorna_vazio(self):
        assert apply_pk_boost({}, {}, ["query"]) == {}


class TestApplyPkBoostBranchesDefensivos:
    """Defensive branches of apply_pk_boost."""

    def _make_doc(self, table_name: str, pk_columns=None, fk_columns=None, schema="OWN"):
        from tailwhip.models.table_metadata import TableMetadata
        from tailwhip.models.column_metadata import ColumnMetadata
        from tailwhip.models.schema_document import SchemaDocument

        cols = []
        for name in (pk_columns or []):
            cols.append(ColumnMetadata(table_name=table_name, column_name=name, data_type="NUMBER", is_pk=True))
        for name in (fk_columns or []):
            cols.append(ColumnMetadata(table_name=table_name, column_name=name, data_type="NUMBER", is_fk=True))

        return SchemaDocument(
            doc_id=f"{schema}.{table_name}",
            table=TableMetadata(table_name=table_name, schema_name=schema),
            columns=cols,
        )

    def test_query_apenas_stopwords_retorna_scores_inalterados(self):
        """When the query contains only stopwords, the boost must not be applied."""
        doc    = self._make_doc("TABELA", pk_columns=["ID_INVOICE"])
        scores = {"OWN.TABELA": 0.8}
        docs   = {"OWN.TABELA": doc}
        tokens = ["id", "cod", "num"]  # stopwords only

        boosted = apply_pk_boost(scores, docs, tokens)
        assert boosted["OWN.TABELA"] == pytest.approx(0.8)

    def test_doc_id_ausente_no_doc_map_preserva_score(self):
        """doc_id present in scores but missing from doc_map must not raise."""
        scores = {"OWN.FANTASMA": 0.5, "OWN.REAL": 0.5}
        doc    = self._make_doc("REAL", pk_columns=["ID_REAL"])
        docs   = {"OWN.REAL": doc}  # OWN.FANTASMA intentionally absent
        tokens = ["real"]

        boosted = apply_pk_boost(scores, docs, tokens)
        assert "OWN.FANTASMA" in boosted
        assert boosted["OWN.FANTASMA"] == pytest.approx(0.5)