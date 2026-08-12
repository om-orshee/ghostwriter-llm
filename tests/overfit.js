const Transformer = require('../transformer');
const { BPETokenizer } = require('../tokenizer');

// Match tokenizer's effective vocab size (256 base + 30 merges = 286)
const tok = new BPETokenizer(286);
tok.train('The quick brown fox jumps over the lazy dog. '.repeat(20), 30);

// Increase lr from 0.001 → 0.05, reduce layers for faster convergence on tiny data
const model = new Transformer(0.05, 286, 128, 4, 4, 512, 128);

const tokens = tok.encode(
  'The quick brown fox jumps over the lazy dog. '.repeat(20),
);
const input = new Int32Array(tokens.slice(0, 40));
const target = new Int32Array(tokens.slice(1, 41));

console.log('Initial:', model.forward(input, target).loss.toFixed(4));

for (let i = 0; i < 50; i++) {
  model.zeroGrad();
  const { loss, cache } = model.forward(input, target);
  model.backward(cache);
  model.clipGrads();
  model.update();
  if (i % 10 === 0) console.log('Step', i, 'loss:', loss.toFixed(4));
}

console.log('Final:', model.forward(input, target).loss.toFixed(4));
