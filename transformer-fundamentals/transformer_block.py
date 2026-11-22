
import numpy as np

from multi_head_attention import multi_head_attention

# 1e-5 means 0.00001. It prevents division by zero when the variance is zero and improves numerical stability.
'''
LayerNorm rescales each token's vector so its numbers have mean 0 
and standard deviation 1. From the demo, with a 4-number token:
'''
def layer_norm(x, epsilon=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True) # Variance measures how spread out the numbers are around their mean.

    return (x - mean) / np.sqrt(variance + epsilon) # x - mean centers the numbers around zero.  => And np.sqrt(variance + epsilon) calculates a stabilized standard deviation.

# Rectified Linear Unit.
'''
Neural networks need nonlinear activation functions to learn more complex relationships.
Without them, stacking only linear transformations would still be equivalent to one linear transformation.

Attention = communication between tokens.

FFN = further processing within each token.
'''
def relu(x):
    return np.maximum(0, x)

# FFN processes the features of each token independently to produce a more useful representation.
def feed_forward(x, W1, b1, W2, b2):
    hidden = relu(x @ W1 + b1)
    return hidden @ W2 + b2


def transformer_block(X, weights):
    # First residual branch: attention
    normalized = layer_norm(X)

    attention_output, attention_weights = multi_head_attention(
        normalized,
        weights["W_q"],
        weights["W_k"],
        weights["W_v"],
        weights["W_o"],
        num_heads=2,
    )

    X = X + attention_output # Residual connection

    # Second residual branch: feed-forward network
    normalized = layer_norm(X)

    ffn_output = feed_forward(
        normalized,
        weights["W1"],
        weights["b1"],
        weights["W2"],
        weights["b2"],
    )

    output = X + ffn_output # Residual connection

    return output, attention_weights


if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ])

    d_model = 4
    d_ff = 8

    weights = {
        "W_q": np.random.randn(4, 4) * 0.1,
        "W_k": np.random.randn(4, 4) * 0.1,
        "W_v": np.random.randn(4, 4) * 0.1,
        "W_o": np.random.randn(4, 4) * 0.1,
        "W1": np.random.randn(d_model, d_ff) * 0.1,
        "b1": np.zeros(d_ff),
        "W2": np.random.randn(d_ff, d_model) * 0.1,
        "b2": np.zeros(d_model),
    }

    output, attention_weights = transformer_block(X, weights)

    print("Input shape:", X.shape)
    print("Output shape:", output.shape)
    print("\nTransformer output:")
    print(np.round(output, 3))
