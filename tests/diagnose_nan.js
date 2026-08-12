//tests/diagnose_nan.js
const Transformer = require('../transformer');
const { BPETokenizer } = require('../tokenizer');

const tok = new BPETokenizer(512);
tok.train('Hello world this is a test.', 5);

const model = new Transformer(0.001, 512, 128, 4, 4, 512, 128);
const tokens = tok.encode(
  'Hello world this is a test of the transformer model.',
);

// Tiny input: just 4 tokens
const input = new Int32Array(tokens.slice(0, 4));
const target = new Int32Array(tokens.slice(1, 5));

function hasNaN(arr) {
  for (let i = 0; i < arr.length; i++) if (Number.isNaN(arr[i])) return true;
  return false;
}

function check(name, arr) {
  const nan = hasNaN(arr);
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  console.log(
    `${name}: ${nan ? '*** NaN ***' : 'OK'} | min=${min.toFixed(4)} max=${max.toFixed(4)} | len=${arr.length}`,
  );
  return nan;
}

// Manual forward with checks
const T = input.length;
const { vocabSize, dModel, nLayers, nHeads, dHead, dFF } = model;

console.log(
  `T=${T} dModel=${dModel} nLayers=${nLayers} nHeads=${nHeads} dHead=${dHead}`,
);

// Embeddings
const embeds = new Float32Array(T * dModel);
for (let t = 0; t < T; t++) {
  const tok = input[t];
  const pos = Math.min(t, model.maxSeqLen - 1);
  for (let d = 0; d < dModel; d++) {
    embeds[t * dModel + d] =
      model.wte[tok * dModel + d] + model.wpe[pos * dModel + d];
  }
}
if (check('embeds', embeds)) process.exit(1);

let x = embeds;

for (let l = 0; l < nLayers; l++) {
  console.log(`\n--- Layer ${l} ---`);
  const residual1 = new Float32Array(x);

  // LN1
  const ln1 = require('../matrix').layerNormForward(
    x,
    model.ln1_g[l],
    model.ln1_b[l],
  );
  if (check('ln1.y', ln1.y)) process.exit(1);

  // QKV
  const qkv = require('../matrix').matMul(
    ln1.y,
    model.wqkv[l],
    T,
    dModel,
    3 * dModel,
  );
  if (check('qkv', qkv)) process.exit(1);

  // Attention
  const attnOut = new Float32Array(T * dModel);
  for (let h = 0; h < nHeads; h++) {
    const Q = new Float32Array(T * dHead);
    const K = new Float32Array(T * dHead);
    const V = new Float32Array(T * dHead);
    for (let t = 0; t < T; t++) {
      const qkvBase = t * 3 * dModel;
      const qBase = t * dHead;
      for (let d = 0; d < dHead; d++) {
        Q[qBase + d] = qkv[qkvBase + h * dHead + d];
        K[qBase + d] = qkv[qkvBase + dModel + h * dHead + d];
        V[qBase + d] = qkv[qkvBase + 2 * dModel + h * dHead + d];
      }
    }

    const scores = require('../matrix').matMulBT(Q, K, T, T, dHead);
    const scale = 1.0 / Math.sqrt(dHead);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        if (j > i) scores[i * T + j] = -1e9;
        else scores[i * T + j] *= scale;
      }
    }
    if (check(`scores_h${h}`, scores)) process.exit(1);

    const weights = require('../matrix').softmaxRows(scores, T, T);
    if (check(`weights_h${h}`, weights)) process.exit(1);

    const outH = require('../matrix').matMul(weights, V, T, T, dHead);
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < dHead; d++) {
        attnOut[t * dModel + h * dHead + d] = outH[t * dHead + d];
      }
    }
  }
  if (check('attnOut', attnOut)) process.exit(1);

  // Output projection
  const attnProj = require('../matrix').matMul(
    attnOut,
    model.wo[l],
    T,
    dModel,
    dModel,
  );
  if (check('attnProj', attnProj)) process.exit(1);

  for (let i = 0; i < x.length; i++) x[i] = residual1[i] + attnProj[i];
  if (check('after_attn_residual', x)) process.exit(1);

  const residual2 = new Float32Array(x);
  const ln2 = require('../matrix').layerNormForward(
    x,
    model.ln2_g[l],
    model.ln2_b[l],
  );
  if (check('ln2.y', ln2.y)) process.exit(1);

  const ffnPre = require('../matrix').matMul(
    ln2.y,
    model.wup[l],
    T,
    dModel,
    dFF,
  );
  if (check('ffnPre', ffnPre)) process.exit(1);

  const ffnMid = require('../matrix').gelu(ffnPre);
  if (check('ffnMid', ffnMid)) process.exit(1);

  const ffnOut = require('../matrix').matMul(
    ffnMid,
    model.wdown[l],
    T,
    dFF,
    dModel,
  );
  if (check('ffnOut', ffnOut)) process.exit(1);

  for (let i = 0; i < x.length; i++) x[i] = residual2[i] + ffnOut[i];
  if (check('after_ffn_residual', x)) process.exit(1);
}

// Final LN
const lnF = require('../matrix').layerNormForward(x, model.lnf_g, model.lnf_b);
if (check('lnF.y', lnF.y)) process.exit(1);

const logits = require('../matrix').matMul(
  lnF.y,
  model.wlm,
  T,
  dModel,
  vocabSize,
);
if (check('logits', logits)) process.exit(1);

// Loss
const { crossEntropyLoss } = require('../matrix');
for (let t = 0; t < T; t++) {
  const slice = logits.subarray(t * vocabSize, (t + 1) * vocabSize);
  const loss = crossEntropyLoss(slice, target[t]);
  console.log(`loss[t=${t}]: ${loss.toFixed(4)}`);
}

console.log(
  '\nNo NaN found in forward pass. The bug is in backward or update.',
);
