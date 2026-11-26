def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_ids:
        raise ValueError(
            "Relevant IDs must not be empty."
        )

    top_k_ids = set(retrieved_ids[:k])

    relevant_retrieved = (
        # top_k_ids intersection(relevant_ids)
        top_k_ids.intersection(relevant_ids)
    )

    return len(relevant_retrieved) / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    for rank, document_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    results: list[tuple[list[str], set[str]]],
    k: int,
) -> float:
    if not results:
        raise ValueError("No evaluation results.")

    reciprocal_ranks = [
        reciprocal_rank_at_k(
            retrieved,
            relevant,
            k,
        )
        for retrieved, relevant in results
    ]

    return sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )


if __name__ == "__main__":
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    print(
        "Recall@5:",
        recall_at_k(retrieved, relevant, 5)
    )

    print(
        "Reciprocal Rank@5:",
        reciprocal_rank_at_k(
            retrieved, relevant, 5
        )
    )

    evaluation_results = [
        (["A", "B", "C"], {"A"}),
        (["B", "C", "A"], {"C"}),
        (["B", "C", "D", "A"], {"A"}),
    ]

    print(
        "MRR@5:",
        mean_reciprocal_rank(
            evaluation_results, 5
        )
    )