import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path

# -----------------------------
# 1. Configuration
# -----------------------------
torch.manual_seed(42)

BATCH_SIZE = 16 # Train on 16 text sequences at a time
BLOCK_SIZE = 32 # Each training sequence contains 32 characters
D_MODEL = 48 # Each token has a learned vector containing 48 numbers/features. This is the model dimension.
NUM_HEADS = 2 # Use 2 attention heads
NUM_LAYERS = 2 # Stack 2 Transformer blocks
LEARNING_RATE = 3e-3 # [0.003] => The learning rate controls how aggressively the optimizer updates the model's parameters.
TRAIN_STEPS = 400 # Perform 400 training iterations
EVAL_INTERVAL = 50 # Evaluate training and validation loss every 50 optimizer updates.

# (16 X 32) = 512 tokens per batch. Each token is represented by 48 numbers. So, the model processes 512 X 48 numbers per batch.
# mps => Metal Performance Shaders
device = (
    "mps" if torch.backends.mps.is_available() # apple silicon gpu e.g M1, M2, etc.
    else "cuda" if torch.cuda.is_available() # nvidia gpu
    else "cpu"
)

print("Using device:", device)

# -----------------------------
# 2. Small training corpus
# -----------------------------
sentences = [
    "the payment was approved.",
    "the payment was declined.",
    "the customer made a payment.",
    "the bank processed the payment.",
    "the transfer was successful.",
    "the transaction was completed.",
    "the customer checked the balance.",
    "the payment service is running.",
]

validation_sentences = [
    "the transfer was approved.",
    "the customer processed the payment.",
]

train_text = "\n".join(sentences * 100) + "\n" # contains examples used to update the model's weights.
val_text = "\n".join(validation_sentences * 20) + "\n" # contains different sentences that the model will encounter only during evaluation.

# A shared tokenizer must understand the characters
# appearing in either dataset.
text = train_text + val_text


# -----------------------------
# 3. Character-level tokenizer
# -----------------------------
chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

 # contains training token IDs.
train_data = torch.tensor(
    encode(train_text),
    dtype=torch.long
)

# contains validation token IDs.
val_data = torch.tensor(
    encode(val_text),
    dtype=torch.long
)

print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))
print("Vocabulary size:", vocab_size)

# -----------------------------
# 4. Create training batches
# -----------------------------

def get_batch(split="train", fixed=False):
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")

    source = train_data if split == "train" else val_data

    max_start = len(source) - BLOCK_SIZE - 1

    if max_start < 0:
        raise ValueError("Dataset is too short for BLOCK_SIZE.")

    # When fixed=False, we randomly sample positions in the source data to create batches. When fixed=True, we use evenly spaced positions.
    if fixed:
        positions = torch.linspace( # gives evenly spaced numbers, including both ends and turns them into a tensor. e.g torch.linspace(0, 4, steps=5) => tensor([0., 1., 2., 3., 4.])
            0,
            max_start,
            steps=BATCH_SIZE
        ).long()
    else:
        positions = torch.randint(
            0,
            max_start + 1, # That's why it's max_start + 1: so that max_start itself can be picked.
            (BATCH_SIZE,)
        )

    x = torch.stack([
        source[i:i + BLOCK_SIZE]
        for i in positions.tolist()
    ])

    y = torch.stack([
        source[i + 1:i + BLOCK_SIZE + 1]
        for i in positions.tolist()
    ])

    return x.to(device), y.to(device)

# -----------------------------
# 5. Multi-head causal attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert D_MODEL % NUM_HEADS == 0

        self.head_dim = D_MODEL // NUM_HEADS

        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)

        mask = torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)
        )

        self.register_buffer( # state that belongs to the model but is not a learnable parameter. it belongs to the module and is saved and loaded in the state_dict, but it is not a parameter that will be updated during training.
            "mask", # name of th buffer
            mask.view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view( # view is similar to reshape. It returns a new tensor with the same data but a different shape.
                B, T, NUM_HEADS, self.head_dim
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        out = weights @ v

        out = out.transpose(1, 2)
        out = out.contiguous().view(B, T, C)

        return self.proj(out)

# -----------------------------
# 6. Transformer block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(D_MODEL) # LayerNorm rescales each token's vector so its numbers have mean 0 and standard deviation 1. From the demo, with a 4-number token:

        self.attention = MultiHeadAttention()

        self.ln2 = nn.LayerNorm(D_MODEL)

        self.ffn = nn.Sequential( # nn.Sequential is a container that runs a list of layers in order. The output of each layer becomes the input to the next.
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.ReLU(),
            nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

# -----------------------------
# 7. GPT language model
# -----------------------------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__() # This runs nn.Module's setup, so PyTorch tracks every layer assigned below.

        self.token_embedding = nn.Embedding(
            vocab_size, D_MODEL
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE, D_MODEL
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.ln_final = nn.LayerNorm(D_MODEL)
        # nn.Linear takes its arguments in the order (input size, output size):
        self.lm_head = nn.Linear(
            D_MODEL, vocab_size
        )

    def forward(self, idx, targets=None): # defines how data flows through the neural network.
        B, T = idx.shape # B stands for batch size., T stands for sequence length (number of tokens in each sequence).

        token_vectors = self.token_embedding(idx)

        positions = torch.arange( # arrange It creates a sequence of evenly spaced numbers. e.g torch.arange(4) => tensor([0, 1, 2, 3]). These are the position IDs for the four token positions.
            T, device=idx.device # create the positions tensor on the same device as idx.
        )

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors

        x = self.blocks(x)
        x = self.ln_final(x)

        logits = self.lm_head(x) # what character comes after position t?

        loss = None

        # computes one number that measures how wrong the model's predictions are
        if targets is not None:
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0):
        if temperature <= 0: # The logits become inf/nan and sampling crashes with a confusing error
            raise ValueError("temperature must be greater than 0")

        self.eval() # Switches to evaluation mode

        for _ in range(max_new_tokens): # One loop iteration produces one new character.
            idx_cond = idx[:, -BLOCK_SIZE:] # Keeps only the last 32 characters

            logits, _ = self(idx_cond)

            next_logits = logits[:, -1, :]
            next_logits = next_logits / temperature

            probabilities = F.softmax(
                next_logits, dim=-1
            )
            #Randomly picks one character, weighted by the probabilities
            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

        return idx

# -----------------------------
# 7a. Evaluation (no weight updates)
# -----------------------------
#  it wraps the whole function so no gradients are tracked inside it.
@torch.no_grad()
def evaluate(model):
    model.eval()

    results = {}

    for split in ["train", "val"]:
        x, y = get_batch(split, fixed=True)

        logits, loss = model(x, y)

        results[split] = loss.item() # converts that scalar tensor into an ordinary Python number.

    model.train()

    return results

# -----------------------------
# 8. Training and checkpointing
# -----------------------------
if __name__ == "__main__":
    CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = TinyGPT().to(device)

    # An optimizer is the part of training that actually changes the model's weights.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("Vocabulary size:", vocab_size)
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    best_val_loss = float("inf") # Keep track of the lowest validation loss seen so far.
    best_step = None

    model.train()

    for step in range(TRAIN_STEPS):
        x, y = get_batch("train")

        logits, loss = model(x, y) # PyTorch invokes the model's forward() method through its module calling mechanism.

        optimizer.zero_grad(set_to_none=True) # Clears the gradients left over from the previous step.

        # works out, for every one of the 60,601 weights, how much the loss would change if that weight changed slightly.
        loss.backward() # Compute each parameter's share of the blame for the loss. PyTorch computes the gradients of the loss with respect to each parameter using backpropagation. to work out how each of the 60,601 weights contributed

        torch.nn.utils.clip_grad_norm_( # Cap the total gradient size at 1.0
            model.parameters(), 1.0
        )

        optimizer.step() # Updates the weights. Roughly, for each parameter:
        # parameter.data -= LEARNING_RATE * parameter.grad

        # Evaluation checks generalization, but does not update model weights.
        if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS - 1:
            metrics = evaluate(model)

            train_loss = metrics["train"]
            val_loss = metrics["val"]

            train_ppl = math.exp(train_loss) # Perplexity = e^(cross-entropy loss).
            val_ppl = math.exp(val_loss)

            print(
                f"Step {step:03d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Train PPL: {train_ppl:.2f} | "
                f"Val PPL: {val_ppl:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step

                # Save learned weights (and registered buffers) from the best model.
                torch.save(
                    model.state_dict(),
                    CHECKPOINT_DIR / "best_model.pt"
                )

                # Save the tokenizer vocabulary and architecture settings for inference.
                metadata = {
                    "chars": chars,
                    "block_size": BLOCK_SIZE,
                    "d_model": D_MODEL,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                }

                (CHECKPOINT_DIR / "metadata.json").write_text(
                    json.dumps(metadata, indent=2),
                    encoding="utf-8"
                )

                print("  Saved new best checkpoint!")

    print("\nBest validation loss:", best_val_loss)
    print("Best step:", best_step)
    print("Checkpoint saved to:", CHECKPOINT_DIR)

    # -----------------------------
    # 9. Generate new text from the BEST saved model
    # -----------------------------
    # Reload the best checkpoint, not simply the weights at the final step.
    best_state = torch.load(
        CHECKPOINT_DIR / "best_model.pt",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(best_state)
    model.eval()

    prompt = "the payment "

    context = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    generated = model.generate(
        context,
        max_new_tokens=120,
        temperature=0.8
    )

    print("\nGenerated text:\n")
    print(decode(generated[0].tolist()))

    metrics = evaluate(model)

    train_loss = metrics["train"]
    val_loss = metrics["val"]

    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss)

    print("\nBest checkpoint evaluation:")
    print(f"Train loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
    print(f"Val loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    # Print saved parameter/buffer names and their tensor shapes for learning.
    for name, tensor in model.state_dict().items():
        print("STATE_DICT:", name, tensor.shape)
