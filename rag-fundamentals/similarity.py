
import numpy as np


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    dot_product = np.dot(a, b)

    magnitude_a = np.linalg.norm(a) # linear algebra that calculates the magnitude (length) of vector a using the Euclidean norm. E.g √(1² + 2²)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError(
            "Cosine similarity is undefined for zero vectors."
        )

    return dot_product / (magnitude_a * magnitude_b)


a = [1, 2]
b = [2, 4]
c = [-2, 1]

print("A vs B:", cosine_similarity(a, b))
print("A vs C:", cosine_similarity(a, c))
