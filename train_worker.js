const { parentPort } = require('worker_threads');
const Transformer = require('./transformer');

let model = null;

function flatGradients() {
  const all = [
    model.dwte,
    model.dwpe,
    ...model.dln1_g,
    ...model.dln1_b,
    ...model.dwqkv,
    ...model.dwo,
    ...model.dln2_g,
    ...model.dln2_b,
    ...model.dwup,
    ...model.dwdown,
    model.dlnf_g,
    model.dlnf_b,
    model.dwlm,
  ];
  let total = 0;
  for (const a of all) total += a.length;
  const flat = new Float32Array(total);
  let off = 0;
  for (const a of all) {
    flat.set(a, off);
    off += a.length;
  }
  return flat;
}

parentPort.on('message', (msg) => {
  if (msg.type === 'init') {
    model = new Transformer(
      msg.config.lr,
      msg.config.vocabSize,
      msg.config.dModel,
      msg.config.nLayers,
      msg.config.nHeads,
      msg.config.dFF,
      msg.config.maxSeqLen,
      msg.config.beta1,
      msg.config.beta2,
      msg.config.eps,
    );
    const buf = Buffer.from(
      msg.weights.buffer,
      msg.weights.byteOffset,
      msg.weights.byteLength,
    );
    model._setAllWeights(buf);
    parentPort.postMessage({ type: 'ready' });
  }

  if (msg.type === 'weights') {
    if (!model) return;
    model._setAllWeights(msg.weights);
    console.log(
      `[WORKER] Weight sync received, wte[0]=${model.wte[0].toFixed(6)}`,
    );
  }

  if (msg.type === 'compute') {
    model.zeroGrad();
    let totalLoss = 0;
    const count = msg.inputs.length;

    for (let b = 0; b < count; b++) {
      const input = new Int32Array(msg.inputs[b]);
      const target = new Int32Array(msg.targets[b]);
      const { loss, cache } = model.forward(input, target);
      model.backward(cache);
      totalLoss += loss;
    }

    const grads = flatGradients();
    parentPort.postMessage(
      {
        type: 'result',
        loss: totalLoss / count,
        count,
        gradients: grads,
      },
      [grads.buffer],
    );
  }
});
