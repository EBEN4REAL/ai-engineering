
import numpy as np

# Five tokens, each with three embedding dimensions.
embedding_matrix = np.array([
    [0.2, 0.8, 0.1],  # cat
    [0.3, 0.7, 0.2],  # dog
    [0.9, 0.1, 0.4],  # car
    [0.5, 0.4, 0.6],  # tree
    [0.6, 0.3, 0.7],  # bank
])

token_ids = [0, 1]  # Example token IDs corresponding to the tokens above.

# Look up the embedding vector for each token ID.
embeddings = embedding_matrix[token_ids]

print("Embeddings:")
print(embeddings)

print("Shape:", embeddings.shape)
