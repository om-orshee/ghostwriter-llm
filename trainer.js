const { Worker } = require('worker_threads');
const os = require('os');
const path = require('path');

class ParallelTrainer {
  constructor(model, numWorkers = os.cpus().length) {
    this.model = model;
    this.numWorkers = Math.min(numWorkers, os.cpus().length);
    this.workers = [];
    this._initPromise = null;
  }

  async init() {
    if (this._initPromise) return this._initPromise;

    this._initPromise = new Promise((resolve) => {
      let ready = 0;
      const weightsBuf = this.model._allWeights();
      const weights = new Float32Array(
        weightsBuf.buffer,
        weightsBuf.byteOffset,
        weightsBuf.byteLength / 4,
      );
      const config = {
        lr: this.model.lr,
        vocabSize: this.model.vocabSize,
        dModel: this.model.dModel,
        nLayers: this.model.nLayers,
        nHeads: this.model.nHeads,
        dFF: this.model.dFF,
        maxSeqLen: this.model.maxSeqLen,
        beta1: this.model.beta1,
        beta2: this.model.beta2,
        eps: this.model.eps,
      };

      for (let i = 0; i < this.numWorkers; i++) {
        const worker = new Worker(path.resolve(__dirname, 'train_worker.js'));
        const workerObj = { worker, busy: false, resolve: null };

        worker.on('message', (msg) => {
          if (msg.type === 'ready') {
            ready++;
            if (ready === this.numWorkers) resolve();
          } else if (msg.type === 'result') {
            workerObj.busy = false;
            if (workerObj.resolve) {
              workerObj.resolve(msg);
              workerObj.resolve = null;
            }
          }
        });

        worker.on('error', (err) => console.error('Worker error:', err));
        worker.postMessage({ type: 'init', weights, config });
        this.workers.push(workerObj);
      }
    });

    return this._initPromise;
  }

  async trainBatch(inputs, targets) {
    await this.init();

    const n = inputs.length;
    const chunk = Math.max(1, Math.ceil(n / this.numWorkers));
    const jobs = [];
    let w = 0;

    for (let i = 0; i < n; i += chunk) {
      const end = Math.min(i + chunk, n);

      while (this.workers[w].busy) {
        w = (w + 1) % this.numWorkers;
        if (this.workers.every((x) => x.busy)) break;
      }

      const workerObj = this.workers[w];
      workerObj.busy = true;

      const job = new Promise((resolve) => {
        workerObj.resolve = resolve;
        workerObj.worker.postMessage({
          type: 'compute',
          inputs: inputs.slice(i, end).map((x) => Array.from(x)),
          targets: targets.slice(i, end).map((x) => Array.from(x)),
        });
      });

      jobs.push(job);
      w = (w + 1) % this.numWorkers;
    }

    const results = await Promise.all(jobs);

    this.model.zeroGrad();
    let totalLoss = 0;
    let totalCount = 0;

    for (const r of results) {
      totalLoss += r.loss * r.count;
      totalCount += r.count;

      // ── DIAGNOSTIC: sum of absolute gradients from this worker ──
      // let gradSum = 0;
      // for (let i = 0; i < r.gradients.length; i++)
      //   gradSum += Math.abs(r.gradients[i]);
      // console.log(
      //   `[TRAINER] Worker grad sum: ${gradSum.toFixed(2)} | loss: ${r.loss.toFixed(4)}`,
      // );

      this._addGrads(r.gradients);
    }

    this.model.clipGrads();
    this.model.update();

    // Log main thread weight change
    console.log(
      `[TRAINER] Main wte[0] after update: ${this.model.wte[0].toFixed(6)}`,
    );

    // Broadcast updated weights to all workers for next batch
    const weights = this.model._allWeights();
    for (const w of this.workers) {
      w.worker.postMessage({ type: 'weights', weights });
    }

    return totalLoss / totalCount;
  }

  _addGrads(flat) {
    const all = [
      this.model.dwte,
      this.model.dwpe,
      ...this.model.dln1_g,
      ...this.model.dln1_b,
      ...this.model.dwqkv,
      ...this.model.dwo,
      ...this.model.dln2_g,
      ...this.model.dln2_b,
      ...this.model.dwup,
      ...this.model.dwdown,
      this.model.dlnf_g,
      this.model.dlnf_b,
      this.model.dwlm,
    ];
    let off = 0;
    for (const arr of all) {
      for (let i = 0; i < arr.length; i++) {
        arr[i] += flat[off++];
      }
    }
  }

  terminate() {
    for (const w of this.workers) w.worker.terminate();
    this.workers = [];
    this._initPromise = null;
  }
}

module.exports = { ParallelTrainer };
