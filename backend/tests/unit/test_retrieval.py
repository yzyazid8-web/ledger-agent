from collections.abc import Sequence
from pathlib import Path

from ledger_agent.ingestion.chunking import Chunk
from ledger_agent.ingestion.indexer import build_bm25_index
from ledger_agent.retrieval.hybrid import (
    AutoFilterRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from ledger_agent.retrieval.lexical import LexicalRetriever
from ledger_agent.retrieval.query_analysis import extract_entities, filter_cascade
from ledger_agent.retrieval.store import ChunkStore
from ledger_agent.retrieval.types import RetrievedChunk, SearchFilters
from ledger_agent.tools.search_filings import SearchFilingsTool


def make_chunk(chunk_id: str, text: str, ticker: str = "MMM", year: int = 2018) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_name=f"{ticker}_{year}_10K",
        company=ticker,
        ticker=ticker,
        doc_type="10-K",
        fiscal_year=year,
        fiscal_quarter=None,
        page_index=int(chunk_id[-1]),
        chunk_index=0,
        header=f"{ticker} | 10-K | FY{year} | page 1",
        text=text,
    )


def ranking(*ids: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(make_chunk(i, i), 1.0, r) for r, i in enumerate(ids)]


# Repérage des entités


def test_entities_from_names_aliases_and_tickers() -> None:
    assert extract_entities("What was 3M's FY2018 capex?").tickers == ("MMM",)
    assert extract_entities("Did J&J grow in FY22?").tickers == ("JNJ",)
    assert extract_entities("Is JPM profitable?").tickers == ("JPM",)
    # Un ticker en minuscules n'est pas un ticker : « ba » n'est pas Boeing
    assert extract_entities("a ba b").tickers == ()


def test_years_including_quarter_suffix_and_short_form() -> None:
    assert extract_entities("As of FY2023Q1, by how much?").years == (2023,)
    assert extract_entities("Compare FY21 and FY2022").years == (2021, 2022)


def test_filter_cascade_goes_from_precise_to_broad() -> None:
    cascade = filter_cascade("3M capex FY2017 vs FY2018")

    assert cascade == [
        SearchFilters(ticker="MMM", fiscal_year=2018),
        SearchFilters(ticker="MMM"),
        None,
    ]
    assert filter_cascade("Which company grew most?") == [None]


# Fusion et reclassement


def test_rrf_rewards_items_ranked_well_by_both() -> None:
    fused = reciprocal_rank_fusion([ranking("a1", "b2", "c3"), ranking("b2", "d4")])

    assert [r.chunk.chunk_id for r in fused][:2] == ["b2", "a1"]


class StaticRetriever:
    def __init__(self, items: list[RetrievedChunk]) -> None:
        self.items = items
        self.calls: list[SearchFilters | None] = []

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        self.calls.append(filters)
        return self.items[:k]


class LengthReranker:
    """Faux reranker : préfère les textes les plus longs."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(len(p)) for p in passages]


def test_reranker_reorders_fused_candidates() -> None:
    dense = StaticRetriever(ranking("a1", "bbbbbbbbbb2"))
    hybrid = HybridRetriever(dense, None, LengthReranker(), candidates=10)

    assert [r.chunk.chunk_id for r in hybrid.search("q", 2)] == ["bbbbbbbbbb2", "a1"]


def test_auto_filter_falls_back_when_results_are_missing() -> None:
    inner = StaticRetriever(ranking("a1"))
    results = AutoFilterRetriever(inner).search("3M capex FY2018", k=3)

    # Aucun filtre ne donne 3 résultats : on a essayé du plus précis au plus large
    assert inner.calls == [
        SearchFilters(ticker="MMM", fiscal_year=2018),
        SearchFilters(ticker="MMM"),
        None,
    ]
    assert len(results) == 1


def test_auto_filter_reports_the_filters_actually_applied() -> None:
    outcome = AutoFilterRetriever(StaticRetriever(ranking("a1"))).search_with_outcome(
        "3M capex FY2018", k=3
    )

    assert outcome.requested == SearchFilters(ticker="MMM", fiscal_year=2018)
    assert outcome.applied is None
    assert outcome.relaxed


def test_search_tool_builds_page_source_ids_and_flags_no_relaxation() -> None:
    tool = SearchFilingsTool(StaticRetriever(ranking("x5")))

    result = tool.run("capex", ticker="mmm", fiscal_year=2018, k=1)

    (passage,) = result.passages
    assert passage.source_id == "doc:MMM_2018_10K:p6"
    assert passage.page_number == 6
    assert result.applied_filters == {"ticker": "MMM", "fiscal_year": 2018}
    assert not result.relaxed


def test_search_tool_says_when_it_relaxed_the_filters() -> None:
    result = SearchFilingsTool(StaticRetriever(ranking("x5"))).run("capex", ticker="MMM", k=3)

    assert result.requested_filters == {"ticker": "MMM"}
    assert result.applied_filters == {}
    assert result.relaxed


# BM25 filtré


def test_lexical_retriever_applies_metadata_filters(tmp_path: Path) -> None:
    chunks = [
        make_chunk("c1", "capital expenditures purchases of equipment", "MMM", 2018),
        make_chunk("c2", "capital expenditures purchases of equipment", "AMD", 2018),
        make_chunk("c3", "dividends declared per share", "MMM", 2018),
    ]
    build_bm25_index(chunks, tmp_path)
    retriever = LexicalRetriever.load(tmp_path, ChunkStore(chunks))

    unfiltered = retriever.search("capital expenditures", 5)
    filtered = retriever.search("capital expenditures", 5, SearchFilters(ticker="AMD"))

    assert {r.chunk.chunk_id for r in unfiltered} == {"c1", "c2"}
    assert [r.chunk.chunk_id for r in filtered] == ["c2"]
    assert retriever.search("capital", 5, SearchFilters(ticker="NOPE")) == []
