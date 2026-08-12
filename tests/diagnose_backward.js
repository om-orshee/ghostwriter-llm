//tests/diagnose_backward.js
const Transformer = require('../transformer');
const { BPETokenizer } = require('../tokenizer');

const tok = new BPETokenizer(512);
tok.train(
  'Hello world this is a test of the transformer model learning to repeat text.',
  10,
);

const model = new Transformer(0.001, 512, 128, 4, 4, 512, 128);
const tokens = tok.encode(
  'Hello world this is a test of the transformer model learning to repeat text. ',
);

const input = new Int32Array(tokens.slice(0, 4));
const target = new Int32Array(tokens.slice(1, 5));

function check(name, arr) {
  let hasNan = false;
  let hasInf = false;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (Number.isNaN(arr[i])) hasNan = true;
    if (!Number.isFinite(arr[i]) && !Number.isNaN(arr[i])) hasInf = true;
    if (arr[i] < min) min = arr[i];
    if (arr[i] > max) max = arr[i];
  }
  console.log(
    `${name}: ${hasNan ? '*** NaN ***' : ''}${hasInf ? '*** Inf ***' : ''} min=${min.toFixed(4)} max=${max.toFixed(4)}`,
  );
  return hasNan || hasInf;
}

console.log('=== FORWARD ===');
const { loss, cache } = model.forward(input, target);
console.log(`Loss: ${loss}`);

if (Number.isNaN(loss)) {
  console.log('\nNaN in forward. Checking caches...');

  // Check embeddings
  if (check('embeds', cache.embeds)) process.exit(1);

  // Check each layer
  for (let l = 0; l < cache.layers.length; l++) {
    const lc = cache.layers[l];
    console.log(`\n-- Layer ${l} --`);
    if (check('ln1.y', lc.ln1.y)) process.exit(1);
    if (check('qkv', lc.qkv)) process.exit(1);
    if (check('attnWeights', lc.attnWeights)) process.exit(1);
    if (check('attnOut', lc.attnOut)) process.exit(1);
    if (check('attnProj', lc.attnProj)) process.exit(1);
    if (check('ln2.y', lc.ln2.y)) process.exit(1);
    if (check('ffnPre', lc.ffnPre)) process.exit(1);
    if (check('ffnMid', lc.ffnMid)) process.exit(1);
    if (check('ffnOut', lc.ffnOut)) process.exit(1);
  }

  if (check('lnF.y', cache.lnF.y)) process.exit(1);
  if (check('logits', cache.logits)) process.exit(1);
} else {
  console.log('\n=== BACKWARD ===');
  model.backward(cache);

  console.log('\nChecking gradients...');
  if (check('dwte', model.dwte)) process.exit(1);
  if (check('dwpe', model.dwpe)) process.exit(1);
  for (let l = 0; l < model.nLayers; l++) {
    console.log(`\n-- Layer ${l} grads --`);
    if (check('dln1_g', model.dln1_g[l])) process.exit(1);
    if (check('dln1_b', model.dln1_b[l])) process.exit(1);
    if (check('dwqkv', model.dwqkv[l])) process.exit(1);
    if (check('dwo', model.dwo[l])) process.exit(1);
    if (check('dln2_g', model.dln2_g[l])) process.exit(1);
    if (check('dln2_b', model.dln2_b[l])) process.exit(1);
    if (check('dwup', model.dwup[l])) process.exit(1);
    if (check('dwdown', model.dwdown[l])) process.exit(1);
  }
  if (check('dlnf_g', model.dlnf_g)) process.exit(1);
  if (check('dlnf_b', model.dlnf_b)) process.exit(1);
  if (check('dwlm', model.dwlm)) process.exit(1);

  console.log('\n=== UPDATE ===');
  model.update();

  console.log('\nChecking weights after update...');
  if (check('wte', model.wte)) process.exit(1);
  if (check('wpe', model.wpe)) process.exit(1);

  console.log('\n=== SECOND FORWARD ===');
  const { loss: loss2 } = model.forward(input, target);
  console.log(`Loss 2: ${loss2}`);
}
