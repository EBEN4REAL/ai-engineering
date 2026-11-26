"""Run with pytest after copying rag_eval.py beside existing RAG project modules."""
from rag_eval import citation_structure, recall_at_k, reciprocal_rank_at_k, percentile


def test_recall():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, 3) == 2 / 3


def test_mrr():
    assert reciprocal_rank_at_k(["x", "b", "c"], {"b"}, 3) == 0.5


def test_unanswerable_no_recall():
    assert recall_at_k(["a"], set(), 3) is None


def test_invalid_citation():
    assert not citation_structure("Answer [S2]", [{"source": "a.pdf", "page": 1}])["valid"]


def test_valid_citation():
    assert citation_structure("Answer [S1]", [{"source": "a.pdf", "page": 1}])["valid"]

# This means 95% of searches finished within this many milliseconds. Only the slowest 5% took longer.
def test_percentile():
    assert percentile([10, 20, 30, 40, 50], 0.95) == 48.0 #  48.0 here means 95% of requests finished within about 48 ms.

'''
p	    Name	                 Means
0.0	    min	                     the fastest request
0.5	    50 / median.             the typical request
0.9	    p90	90% are              this fast or faster
0.95	p95	95% are              this fast or faster
0.99	p99	99% are              this fast or faster
1.0	    max                      the slowest request
'''