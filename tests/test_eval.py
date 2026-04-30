from src.eval import QueryEval, aggregate, mrr_at_k, ndcg_at_k, recall_at_k


def _qe(retrieved: list[str], relevant: set[str]) -> QueryEval:
    return QueryEval(qid="t", relevant=relevant, retrieved=retrieved)


def test_recall_perfect():
    qe = _qe(["a", "b", "c"], {"a", "b"})
    assert recall_at_k(qe, 5) == 1.0


def test_recall_partial():
    qe = _qe(["x", "a", "y"], {"a", "b"})
    assert recall_at_k(qe, 3) == 0.5


def test_mrr_first():
    assert mrr_at_k(_qe(["a", "b"], {"a"}), 5) == 1.0


def test_mrr_third():
    assert abs(mrr_at_k(_qe(["x", "y", "a"], {"a"}), 5) - 1 / 3) < 1e-9


def test_ndcg_perfect_singleton():
    qe = _qe(["a"], {"a"})
    assert abs(ndcg_at_k(qe, 1) - 1.0) < 1e-9


def test_aggregate_keys():
    evals = [_qe(["a"], {"a"}), _qe(["x"], {"a"})]
    out = aggregate(evals, ks=(5,))
    assert out["recall@5"] == 0.5
    assert out["hit@5"] == 0.5
