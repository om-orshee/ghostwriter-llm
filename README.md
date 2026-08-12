# Ghostwriter

Smallest possible multi-personality language model built from scratch in pure JavaScript. Zero dependencies. Zero external AI libraries.

## Architecture

- **Base model**: Character-level LSTM (386K parameters)
- **Personality system**: Isolated weight modifiers + short-term history per ghost
- **Training**: Pure Node.js CPU, batching enabled
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
  -d '{"file":"./training/base_corpus.txt","epochs":5,"batchSize":16}'

# 4. Train personalities
curl -X POST http://localhost:3000/train-ghost \
  -H "Content-Type: application/json" \
  -d '{"ghost":"Vera","file":"./training/vera_corpus.txt","epochs":5}'

curl -X POST http://localhost:3000/train-ghost \
  -H "Content-Type: application/json" \
  -d '{"ghost":"Jax","file":"./training/jax_corpus.txt","epochs":5}'

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
epochs: Number of training passes
batchSize: Sequences per weight update (base: 64, ghost: 16)
stride: Window overlap (64 = none, 32 = 50%, 8 = lots)
temperature: Sampling randomness for generation (0.3–1.0)

# Research Notes

Pure JS constraint: No GPU, no SIMD, no external ML libraries
LSTM ceiling: 386K parameters, character-level. Coherent fragments possible, sustained dialogue difficult.
Isolation mechanism: Each ghost clones base weights and trains personal deltas. History arrays are JS-scope isolated.
Future path: Paralellization for faster training on multiple cores.
