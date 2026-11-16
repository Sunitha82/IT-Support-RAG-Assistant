"""Candidate filtering and parent expansion, independent of model downloads."""
from unittest.mock import Mock
from app.retrieval.enhanced_retriever import EnhancedRetriever, expand_it_terms


def candidate(parent="parent-1", child="child-1", distance=0.2, score=2):
    return {"parent_id": parent, "child_id": child, "source": "vpn.txt",
            "content": "Complete parent procedure", "distance": distance,
            "rerank_score": score, "rrf_score": 0.04}


def test_parent_expansion_dedup_and_actual_scores():
    store, ranker = Mock(), Mock()
    store.multi_query_search.return_value = [candidate(), candidate(child="child-2")]
    ranker.rerank.return_value = store.multi_query_search.return_value
    result = EnhancedRetriever(store, ranker).retrieve("VPN help")
    assert len(result) == 1
    assert result[0].chunk_id == "parent-1"
    assert result[0].child_id == "child-1"
    assert result[0].content == "Complete parent procedure"
    assert result[0].rerank_score == 2
    assert result[0].rrf_score == 0.04


def test_low_similarity_and_negative_rerank_are_rejected():
    store, ranker = Mock(), Mock()
    store.multi_query_search.return_value = [candidate()]
    ranker.rerank.return_value = [candidate(distance=0.9), candidate(score=-1)]
    assert EnhancedRetriever(store, ranker).retrieve("unrelated") == []


def test_expansion_is_explicit_and_used_for_reranking():
    question = "My secure work connection says connected, but company websites won't open."
    expanded = expand_it_terms(question)
    assert "VPN" in expanded and "internal applications" in expanded
    store, ranker = Mock(), Mock()
    store.multi_query_search.return_value = [candidate()]
    ranker.rerank.return_value = [candidate()]
    EnhancedRetriever(store, ranker).retrieve(question)
    assert expanded in store.multi_query_search.call_args.kwargs["additional_queries"]
    assert ranker.rerank.call_args.args[0] == expanded
