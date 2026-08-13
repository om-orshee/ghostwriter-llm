# Ghostwriter - LLM

Smallest possible multi-personality language model built from scratch in pure JavaScript. Zero dependencies. Zero external AI libraries.

## Architecture

- **Base model**: Decoder-only transformer (~4.3M parameters)
  - 256 dModel, 6 layers, 8 attention heads, 768 FFN dim
  - Causal self-attention with KV-cache for fast generation
  - Pre-LayerNorm, GELU activations, learned positional embeddings
- **Tokenizer**: Byte-level BPE (512 vocab size), trained on your corpus
- **Personality system**: Isolated weight clones + short-term history per ghost
- **Training**: Multi-core via Node.js `worker_threads` (data-parallel batches)
- **API**: HTTP endpoints for training and conversation

## Quick Start

```bash
# 1. Generate training data
node generate_synthetic_data.js

# 2. Start server
node server.js

# 3. Train tokenizer + base model
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{"file":"./output/training/base_corpus.txt","epochs":5,"batchSize":16}'

# 4. Train personalities
curl -X POST http://localhost:3000/train-ghost \
  -H "Content-Type: application/json" \
  -d '{"ghost":"Vera","file":"./output/training/vera_corpus.txt","epochs":5}'

curl -X POST http://localhost:3000/train-ghost \
  -H "Content-Type: application/json" \
  -d '{"ghost":"Jax","file":"./output/training/jax_corpus.txt","epochs":5}'

# 5. Speak
curl -X POST http://localhost:3000/speak \
  -H "Content-Type: application/json" \
  -d '{"ghost":"Vera","message":"Hello"}'
```

| Endpoint              | Method   | Description                       |
| --------------------- | -------- | --------------------------------- |
| `/`                   | GET      | Info and available endpoints      |
| `/train`              | POST     | Train base model                  |
| `/train-ghost`        | POST     | Train personality weights         |
| `/speak`              | POST     | Talk to a ghost                   |
| `/ghosts`             | GET/POST | List or create ghosts             |
| `/history?ghost=NAME` | GET      | View ghost's conversation history |

# Training Parameters

file: Path to training text file
text: Inline training text (alternative to file)
epochs: Number of training passes
batchSize: Sequences per gradient step (base: 16, ghost: 8–16)
stride: Window overlap (64 = none, 32 = 50%, 8 = lots)
temperature: Sampling randomness for generation (0.3–1.0)

# Architecture Details

| Component        | Size                 |
| ---------------- | -------------------- |
| Vocabulary       | 512 (byte-level BPE) |
| Embedding dim    | 256                  |
| Layers           | 6                    |
| Attention heads  | 8 (32 dim per head)  |
| FFN hidden dim   | 768                  |
| Max context      | 256 tokens           |
| **Total params** | **~4.3M**            |

# How Personalities Work

Each ghost receives a clone of the base model weights at creation time. When you train a ghost:

1. Its personal model is fine-tuned on ghost-specific text
2. Only that ghost's weights are updated — others are isolated
3. Conversation history is stored per-ghost in memory (not shared)
4. Weights and history are saved to disk on SIGINT and after each /speak

# Performance Notes

Pure JS means no GPU, no SIMD, no BLAS. Training is CPU-bound.
Multi-core training uses worker_threads to parallelize batch gradient computation.
On a modern desktop (e.g. i9, 16+ cores), expect 2–10 seconds per batch depending on batch size.
For faster iteration during testing, reduce the model in server.js:

```js
## new Transformer(0.001, 512, 128, 4, 4, 512, 128); // ~1.1M params
```

# Research Notes

Isolation mechanism: Weight cloning at ghost creation + independent .train() calls. No LoRA or adapters — full parameter deltas.

Tokenizer: Trained once during first /train call, then saved to tokenizer.json. Reused for all ghosts.

Generation: KV-cache avoids recomputing past keys/values, making token-by-token generation roughly linear in sequence length instead of quadratic.

Future path: Quantization (int8 weights), larger context via RoPE or ALiBi, or a tiny WASM SIMD kernel for matmul.
