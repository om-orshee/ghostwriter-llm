//tests/test_overfit.js
const Transformer = require('../transformer');
const { BPETokenizer } = require('../tokenizer');

const tok = new BPETokenizer(512);
tok.train(
  'Hello world this is a test of the transformer model learning to repeat text.',
  10,
);

const model = new Transformer(0.01, 512, 128, 4, 4, 512, 128);
const tokens = tok.encode(
  'Hello world this is a test of the transformer model learning to repeat text. ',
);

// Single sequence
const input = new Int32Array(tokens.slice(0, 64));
const target = new Int32Array(tokens.slice(1, 65));

console.log('Initial loss:', model.forward(input, target).loss.toFixed(4));

for (let i = 0; i < 50; i++) {
  const { loss, cache } = model.forward(input, target);
  model.backward(cache);
  model.clipGrads();
  model.update();
  if (i % 10 === 0) console.log(`Step ${i}: loss = ${loss.toFixed(4)}`);
}

console.log('Final loss:', model.forward(input, target).loss.toFixed(4));
