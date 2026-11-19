
import numpy as np

# axis=-1 adds the numbers across each row:
# keepdims=True keeps the dimensions of the array the same after summing.
# Softmax converts a group of numerical scores into nonnegative weights that add up to 1.
# To improve numerical stability by avoiding unnecessarily large exponentials.
def softmax(x):
    print("Softmax input:", x)
    print("MAX =>", np.max(x, axis=-1, keepdims=True))
    x = x - np.max(x, axis=-1, keepdims=True)
    print("Softmax after subtracting max:", x)
    exp_x = np.exp(x)
    print("Softmax exponentials:", exp_x)

    result = exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )
    print("Softmax result:", result)

    return result


# Three tokens, each with four dimensions.
X = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
])

# Learned projection matrices (identity for illustration).
W_q = np.eye(4) 
W_k = np.eye(4)
W_v = np.eye(4)

print("W_q =>", W_q)

# Step 1: Generate queries, keys, and values.
Q = X @ W_q
K = X @ W_k
V = X @ W_v

# Step 2: Calculate attention scores.
# K.shape[-1] returns the size of the last dimension of K, which is the embedding dimension (4 in this case).
d_k = K.shape[-1]

# K.T is the transpose of K, which allows us to compute the dot product between Q and K for all token pairs.
#K.T transposes the matrix, meaning it swaps rows and columns.

scores = (Q @ K.T) / np.sqrt(d_k)

# Step 3: Prevent attention to future positions.
# np.ones_like(scores) creates an array of ones having the same shape as scores.
# dtype=bool makes the array a boolean array, where True represents positions to be masked.
mask = np.triu(
    np.ones_like(scores, dtype=bool),
    k=1  # k=1 means the diagonal and above will be True (masked), while below the diagonal will be False (unmasked).
)

print("Mask:", mask)

'''
If mask is True:
    use negative infinity
Otherwise:
    keep the original score
'''
scores = np.where(mask, -np.inf, scores)

# Step 4: Convert scores to attention weights.
weights = softmax(scores)

# Step 5: Compute context-aware output vectors.
output = weights @ V

print("Attention weights:")
print(np.round(weights, 3))

print("\nAttention output:")
print(np.round(output, 3))
