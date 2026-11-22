
import torch
import torch.nn as nn
import torch.nn.functional as F

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

text = "\n".join(sentences * 100) + "\n"

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

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

# -----------------------------
# 4. Create training batches
# -----------------------------
def get_batch():
    positions = torch.randint( # torch.randint(low, high, size)
        0,
        len(data) - BLOCK_SIZE, # becuase we want to reserve the last BLOCK_SIZE tokens for the target sequence. so that there is always a next token to predict for each token in the input sequence which we represent with 
        (BATCH_SIZE,) # One-dimensional shape
    )

    x = torch.stack([
        data[i:i + BLOCK_SIZE]
        for i in positions.tolist()
    ])

    y = torch.stack([
        data[i + 1:i + BLOCK_SIZE + 1]
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
# 8. Training
# -----------------------------
model = TinyGPT().to(device) 

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Vocabulary size:", vocab_size)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

model.train()

for step in range(TRAIN_STEPS):
    x, y = get_batch()

    logits, loss = model(x, y) # PyTorch invokes the model's forward() method through its module calling mechanism.

    optimizer.zero_grad(set_to_none=True) # Clears the gradients left over from the previous step.

    loss.backward() # Compute each parameter's share of the blame for the loss. PyTorch computes the gradients of the loss with respect to each parameter using backpropagation.

    torch.nn.utils.clip_grad_norm_( # Cap the total gradient size at 1.0
        model.parameters(), 1.0
    )

    optimizer.step() # Updates the weights. Roughly, for each parameter:
    # parameter.data -= LEARNING_RATE * parameter.grad

    if step % 50 == 0 or step == TRAIN_STEPS - 1:
        print(
            f"Step {step:03d} | Loss: {loss.item():.4f}"
        )

# -----------------------------
# 9. Generate new text
# -----------------------------
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
