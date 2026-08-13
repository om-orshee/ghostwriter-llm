//migrate.js
const fs = require('fs');

const moves = [
  ['training/tokenizer.json', 'output/tokenizer.json'],
  ['weights.bin', 'output/weights.bin'],
  ['ghosts', 'output/ghosts'],
  ['checkpoints', 'output/checkpoints'],
  ['training/base_corpus.txt', 'output/training/base_corpus.txt'],
  ['training/vera_corpus.txt', 'output/training/vera_corpus.txt'],
  ['training/jax_corpus.txt', 'output/training/jax_corpus.txt'],
];

for (const d of ['output', 'output/training', 'output/ghosts', 'output/checkpoints']) {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
}

for (const [src, dest] of moves) {
  if (!fs.existsSync(src)) continue;
  try {
    fs.renameSync(src, dest);
    console.log(`✓ ${src} → ${dest}`);
  } catch (e) {
    console.error(`✗ ${src}: ${e.message}`);
  }
}

console.log('\nDone. You can delete this script.');