
# import numpy as np

# '''
# NOTE => Rows correspond to query tokens. Columns correspond to key tokens.
# '''

# def softmax(x):
#     x = x - np.max(x, axis=-1, keepdims=True)
#     exp_x = np.exp(x)

#     return exp_x / np.sum(
#         exp_x, axis=-1, keepdims=True
#     )


# def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads):
#     seq_len, d_model = X.shape

#     assert d_model % num_heads == 0

#     head_dim = d_model // num_heads

#     # 1. Linear projections
#     Q = X @ W_q
#     K = X @ W_k
#     V = X @ W_v

#     # 2. Split into attention heads
#     Q = Q.reshape(seq_len, num_heads, head_dim) # reshape() changes how the elements of an array are organized into dimensions. It does not change the values or their order in this example.
#     K = K.reshape(seq_len, num_heads, head_dim)
#     V = V.reshape(seq_len, num_heads, head_dim)

#     print("Q shape:", Q)

#     # Shape: (heads, tokens, head_dim) => Bring all the heads to the front for easier matrix multiplication. All the heads would be collected in each row of the new array, and each row would contain all the tokens for that head.
#     Q = Q.transpose(1, 0, 2)
#     K = K.transpose(1, 0, 2)
#     V = V.transpose(1, 0, 2)

#     print("Q shape after transpose:", Q) # This is to ensure that data is grouped by head so each head is now on the rows

#     # 3. Compute scaled dot-product scores
#     # K.T is the transpose of K, which allows us to compute the dot product between Q and K for all token pairs.
#     # Remember the rule of multiplication, when multiplt matrices, the inner dimensions must match. The inner dimension of Q is head_dim, and the inner dimension of K.T is also head_dim. So they can be multiplied together.
#     # (A x B) @ (B x C) = (A x C)
#     scores = Q @ K.transpose(0, 2, 1)
#     print("Scores shape before scaling:", scores.shape)
#     scores = scores / np.sqrt(head_dim)

#     print("Scores shape:", scores)

#     # 4. Causal mask: block future tokens
#     mask = np.triu(
#         np.ones((seq_len, seq_len), dtype=bool),
#         k=1
#     )

#     print("Mask shape:", mask.shape)
#     # None -> Insert a new dimension at the beginning
#     # First : -> 
#     scores = np.where(mask[None, :, :], -np.inf, scores)

#     # 5. Normalise attention scores
#     weights = softmax(scores)

#     # 6. Combine value vectors
#     heads = weights @ V

#     # 7. Concatenate heads
#     combined = heads.transpose(1, 0, 2)
#     combined = combined.reshape(seq_len, d_model)

#     # 8. Output projection
#     output = combined @ W_o

#     return output, weights


# if __name__ == "__main__":
#     # This initializes NumPy's legacy random number generator with seed 42.
#     np.random.seed(42)
#     print("Random seed set to 42 for reproducibility.", np.random.seed(42))
#     X = np.array([
#         [1.0, 0.0, 0.0, 1.0],
#         [0.0, 1.0, 0.0, 1.0],
#         [0.0, 0.0, 1.0, 1.0],
#     ])
#     # head_dim = d_model // num_heads
#     d_model = 4 # model dimension (means each input token is represented by four numbers.)
#     num_heads = 2 # means we want two attention heads.

#     W_q = np.random.randn(4, 4) * 0.1 # Multiplying by 0.1 makes those numbers smaller.
#     W_k = np.random.randn(4, 4) * 0.1
#     W_v = np.random.randn(4, 4) * 0.1
#     W_o = np.random.randn(4, 4) * 0.1

#     output, weights = multi_head_attention(
#         X, W_q, W_k, W_v, W_o, num_heads
#     )

#     print("Output shape:", output.shape)
#     print("Attention weights shape:", weights.shape)

#     print("\nHead 1 weights:")
#     print(np.round(weights[0], 3))

#     print("\nHead 2 weights:")
#     print(np.round(weights[1], 3))

#     print("\nFinal output:")
#     print(np.round(output, 3))




import numpy as np

np.set_printoptions(precision=5, suppress=True)


def show(name, value):
    print(f"\n{'=' * 55}")
    print(name)
    print(f"Shape: {value.shape}")
    print(value)


def softmax(x):
    show("Softmax input", x)

    row_max = np.max(x, axis=-1, keepdims=True)
    show("Maximum per row", row_max)

    shifted = x - row_max
    show("Scores after subtracting maximum", shifted)

    exp_x = np.exp(shifted)
    show("Exponentials", exp_x)

    denominator = np.sum(
        exp_x, axis=-1, keepdims=True
    )
    show("Sum of exponentials", denominator)

    result = exp_x / denominator
    show("Final softmax probabilities", result)

    return result


def multi_head_attention(
    X, W_q, W_k, W_v, W_o, num_heads
):
    show("Input X", X)

    seq_len, d_model = X.shape
    print(f"\nseq_len = {seq_len}")
    print(f"d_model = {d_model}")
    print(f"num_heads = {num_heads}")

    assert d_model % num_heads == 0

    head_dim = d_model // num_heads
    print(f"head_dim = {head_dim}")

    # 1. Linear projections
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    show("Q after projection", Q)
    show("K after projection", K)
    show("V after projection", V)

    # 2. Split features into heads
    Q = Q.reshape(seq_len, num_heads, head_dim)
    K = K.reshape(seq_len, num_heads, head_dim)
    V = V.reshape(seq_len, num_heads, head_dim)

    show("Q after reshape", Q)
    show("K after reshape", K)
    show("V after reshape", V)

    # 3. Group data by attention head
    Q = Q.transpose(1, 0, 2)
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    show("Q after transpose", Q)
    show("K after transpose", K)
    show("V after transpose", V)

    # 4. Calculate raw attention scores
    K_transposed = K.transpose(0, 2, 1)
    show("K with last two axes swapped", K_transposed)

    scores = Q @ K_transposed
    show("Raw Q @ K_transposed scores", scores)

    # 5. Scale
    scale = np.sqrt(head_dim)
    print(f"\nScaling factor: {scale}")

    scores = scores / scale
    show("Scaled attention scores", scores)

    # 6. Create causal mask
    initial_mask = np.ones(
        (seq_len, seq_len), dtype=bool
    )
    show("Initial Boolean mask", initial_mask)

    mask = np.triu(initial_mask, k=1)
    show("Causal mask", mask)

    expanded_mask = mask[None, :, :]
    show("Expanded mask", expanded_mask)

    scores = np.where(
        expanded_mask, -np.inf, scores
    )
    show("Masked attention scores", scores)

    # 7. Convert scores into weights
    weights = softmax(scores)
    show("Attention weights", weights)

    # 8. Weighted sum of values
    heads = weights @ V
    show("Contextualized attention heads", heads)

    # 9. Reorganize and concatenate heads
    heads_by_token = heads.transpose(1, 0, 2)
    show("Heads regrouped by token", heads_by_token)

    combined = heads_by_token.reshape(
        seq_len, d_model
    )
    show("Concatenated head outputs", combined)

    # 10. Final output projection
    output = combined @ W_o
    show("Final projected output", output)

    return output, weights


if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])

    d_model = 4
    num_heads = 2

    W_q = np.random.randn(4, 4) * 0.1
    W_k = np.random.randn(4, 4) * 0.1
    W_v = np.random.randn(4, 4) * 0.1
    W_o = np.random.randn(4, 4) * 0.1

    show("W_q", W_q)
    show("W_k", W_k)
    show("W_v", W_v)
    show("W_o", W_o)

    output, weights = multi_head_attention(
        X, W_q, W_k, W_v, W_o, num_heads
    )

    print("\nFinal output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
