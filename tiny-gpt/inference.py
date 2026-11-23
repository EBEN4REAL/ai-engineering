
import json
from pathlib import Path

import torch

from train import (
    TinyGPT,
    BLOCK_SIZE,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    chars as training_chars,
    device,
)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# 1. Read checkpoint metadata.
metadata = json.loads(
    (CHECKPOINT_DIR / "metadata.json").read_text(
        encoding="utf-8"
    )
)

# 2. Verify architectural compatibility.
assert metadata["block_size"] == BLOCK_SIZE
assert metadata["d_model"] == D_MODEL
assert metadata["num_heads"] == NUM_HEADS
assert metadata["num_layers"] == NUM_LAYERS
assert metadata["chars"] == training_chars

# 3. Restore the tokenizer.
chars = metadata["chars"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(text):
    unknown = sorted(set(text) - set(stoi))

    if unknown:
        raise ValueError(
            f"Characters not in vocabulary: {unknown}"
        )

    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


# 4. Reconstruct the model architecture.
model = TinyGPT().to(device)

# 5. Restore trained parameters.
state_dict = torch.load(
    CHECKPOINT_DIR / "best_model.pt",
    map_location=device,
    weights_only=True, # resteict reading to model weights only, ignore optimizer state and other metadata for security and compatibility reasons
)

model.load_state_dict(state_dict)

# 6. Switch to inference mode.
model.eval()

print("Model loaded successfully!")
print("Best training step:", metadata["best_step"])
print("Best validation loss:", metadata["best_val_loss"])

# 7. Encode the input prompt.
prompt = "the payment "

context = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device,
)

# 8. Generate without recording gradients.
# with is like a cleaup block, it will automatically clean up the context after the block is executed. with uses a context manager to manage resources, in this case, it will disable gradient calculation and free up memory after the block is executed.
with torch.inference_mode():
    generated = model.generate(
        context,
        max_new_tokens=120,
        temperature=0.8, # temperature=0.8 makes the model slightly more likely to choose high-probability tokens while still allowing some variety.
    )

# 9. Decode generated token IDs.
# print("Generated token IDs:", generated[0].tolist())
output = decode(generated[0].tolist())

print("\nGenerated text:\n")
print(output)
