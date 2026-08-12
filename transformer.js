// transformer.js — Small decoder-only transformer in pure Node.js
const fs = require('fs');
const {
  zeros,
  randnInit,
  matMul,
  matMulAT,
  matMulBT,
  addInPlace,
  scaleInPlace,
  softmaxRows,
  layerNormForward,
  layerNormBackward,
  gelu,
  geluDeriv,
  crossEntropyLoss,
  crossEntropyBackward,
  sample,
} = require('./matrix');

class Transformer {
  constructor(
    learningRate = 0.001,
    vocabSize = 512,
    dModel = 256,
    nLayers = 6,
    nHeads = 8,
    dFF = 768,
    maxSeqLen = 256,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
  ) {
    this.lr = learningRate;
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.nLayers = nLayers;
    this.nHeads = nHeads;
    this.dHead = Math.floor(dModel / nHeads);
    this.dFF = dFF;
    this.maxSeqLen = maxSeqLen;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0;

    // Embeddings
    this.wte = zeros(vocabSize * dModel);
    this.wpe = zeros(maxSeqLen * dModel);

    // Per-layer weights
    this.ln1_g = [];
    this.ln1_b = [];
    this.wqkv = [];
    this.wo = [];
    this.ln2_g = [];
    this.ln2_b = [];
    this.wup = [];
    this.wdown = [];

    for (let l = 0; l < nLayers; l++) {
      this.ln1_g.push(zeros(dModel));
      this.ln1_b.push(zeros(dModel));
      this.wqkv.push(zeros(dModel * 3 * dModel));
      this.wo.push(zeros(dModel * dModel));
      this.ln2_g.push(zeros(dModel));
      this.ln2_b.push(zeros(dModel));
      this.wup.push(zeros(dModel * dFF));
      this.wdown.push(zeros(dFF * dModel));
    }

    this.lnf_g = zeros(dModel);
    this.lnf_b = zeros(dModel);
    this.wlm = zeros(dModel * vocabSize);

    // Gradients
    this.dwte = zeros(vocabSize * dModel);
    this.dwpe = zeros(maxSeqLen * dModel);
    this.dln1_g = [];
    this.dln1_b = [];
    this.dwqkv = [];
    this.dwo = [];
    this.dln2_g = [];
    this.dln2_b = [];
    this.dwup = [];
    this.dwdown = [];
    for (let l = 0; l < nLayers; l++) {
      this.dln1_g.push(zeros(dModel));
      this.dln1_b.push(zeros(dModel));
      this.dwqkv.push(zeros(dModel * 3 * dModel));
      this.dwo.push(zeros(dModel * dModel));
      this.dln2_g.push(zeros(dModel));
      this.dln2_b.push(zeros(dModel));
      this.dwup.push(zeros(dModel * dFF));
      this.dwdown.push(zeros(dFF * dModel));
    }
    this.dlnf_g = zeros(dModel);
    this.dlnf_b = zeros(dModel);
    this.dwlm = zeros(dModel * vocabSize);

    // Adam first and second moments
    this.mwte = zeros(vocabSize * dModel);
    this.vwte = zeros(vocabSize * dModel);
    this.mwpe = zeros(maxSeqLen * dModel);
    this.vwpe = zeros(maxSeqLen * dModel);

    this.mln1_g = [];
    this.vln1_g = [];
    this.mln1_b = [];
    this.vln1_b = [];
    this.mwqkv = [];
    this.vwqkv = [];
    this.mwo = [];
    this.vwo = [];
    this.mln2_g = [];
    this.vln2_g = [];
    this.mln2_b = [];
    this.vln2_b = [];
    this.mwup = [];
    this.vwup = [];
    this.mwdown = [];
    this.vwdown = [];

    for (let l = 0; l < nLayers; l++) {
      this.mln1_g.push(zeros(dModel));
      this.vln1_g.push(zeros(dModel));
      this.mln1_b.push(zeros(dModel));
      this.vln1_b.push(zeros(dModel));
      this.mwqkv.push(zeros(dModel * 3 * dModel));
      this.vwqkv.push(zeros(dModel * 3 * dModel));
      this.mwo.push(zeros(dModel * dModel));
      this.vwo.push(zeros(dModel * dModel));
      this.mln2_g.push(zeros(dModel));
      this.vln2_g.push(zeros(dModel));
      this.mln2_b.push(zeros(dModel));
      this.vln2_b.push(zeros(dModel));
      this.mwup.push(zeros(dModel * dFF));
      this.vwup.push(zeros(dModel * dFF));
      this.mwdown.push(zeros(dFF * dModel));
      this.vwdown.push(zeros(dFF * dModel));
    }

    this.mlnf_g = zeros(dModel);
    this.vlnf_g = zeros(dModel);
    this.mlnf_b = zeros(dModel);
    this.vlnf_b = zeros(dModel);
    this.mwlm = zeros(dModel * vocabSize);
    this.vwlm = zeros(dModel * vocabSize);

    this.initWeights();
  }

  initWeights() {
    const sEmbed = Math.sqrt(2.0 / this.vocabSize);
    for (let i = 0; i < this.wte.length; i++)
      this.wte[i] = (Math.random() * 2 - 1) * sEmbed;
    for (let i = 0; i < this.wpe.length; i++)
      this.wpe[i] = (Math.random() * 2 - 1) * 0.01;

    for (let l = 0; l < this.nLayers; l++) {
      this.ln1_g[l].fill(1.0);
      this.ln2_g[l].fill(1.0);
      this.wqkv[l].set(randnInit(this.wqkv[l].length, this.dModel));
      this.wo[l].set(randnInit(this.wo[l].length, this.dModel));
      this.wup[l].set(randnInit(this.wup[l].length, this.dModel));
      this.wdown[l].set(randnInit(this.wdown[l].length, this.dFF));
    }
    this.lnf_g.fill(1.0);
    this.wlm.set(randnInit(this.wlm.length, this.dModel));
  }

  paramCount() {
    let total =
      this.wte.length +
      this.wpe.length +
      this.lnf_g.length +
      this.lnf_b.length +
      this.wlm.length;
    for (let l = 0; l < this.nLayers; l++) {
      total +=
        this.ln1_g[l].length +
        this.ln1_b[l].length +
        this.wqkv[l].length +
        this.wo[l].length;
      total +=
        this.ln2_g[l].length +
        this.ln2_b[l].length +
        this.wup[l].length +
        this.wdown[l].length;
    }
    return total;
  }

  zeroGrad() {
    this.dwte.fill(0);
    this.dwpe.fill(0);
    for (let l = 0; l < this.nLayers; l++) {
      this.dln1_g[l].fill(0);
      this.dln1_b[l].fill(0);
      this.dwqkv[l].fill(0);
      this.dwo[l].fill(0);
      this.dln2_g[l].fill(0);
      this.dln2_b[l].fill(0);
      this.dwup[l].fill(0);
      this.dwdown[l].fill(0);
    }
    this.dlnf_g.fill(0);
    this.dlnf_b.fill(0);
    this.dwlm.fill(0);
  }

  clipGrads(max = 1.0) {
    const all = [
      this.dwte,
      this.dwpe,
      ...this.dln1_g,
      ...this.dln1_b,
      ...this.dwqkv,
      ...this.dwo,
      ...this.dln2_g,
      ...this.dln2_b,
      ...this.dwup,
      ...this.dwdown,
      this.dlnf_g,
      this.dlnf_b,
      this.dwlm,
    ];
    for (const g of all) {
      for (let i = 0; i < g.length; i++) {
        if (g[i] > max) g[i] = max;
        else if (g[i] < -max) g[i] = -max;
      }
    }
  }

  update() {
    this.t++;
    const b1t = Math.pow(this.beta1, this.t);
    const b2t = Math.pow(this.beta2, this.t);
    const lr = (this.lr * Math.sqrt(1 - b2t)) / (1 - b1t);

    const adamStep = (w, dw, m, v) => {
      for (let i = 0; i < w.length; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * dw[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * dw[i] * dw[i];
        const mHat = m[i] / (1 - b1t);
        const vHat = v[i] / (1 - b2t);
        w[i] -= (lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    };

    adamStep(this.wte, this.dwte, this.mwte, this.vwte);
    adamStep(this.wpe, this.dwpe, this.mwpe, this.vwpe);
    for (let l = 0; l < this.nLayers; l++) {
      adamStep(this.ln1_g[l], this.dln1_g[l], this.mln1_g[l], this.vln1_g[l]);
      adamStep(this.ln1_b[l], this.dln1_b[l], this.mln1_b[l], this.vln1_b[l]);
      adamStep(this.wqkv[l], this.dwqkv[l], this.mwqkv[l], this.vwqkv[l]);
      adamStep(this.wo[l], this.dwo[l], this.mwo[l], this.vwo[l]);
      adamStep(this.ln2_g[l], this.dln2_g[l], this.mln2_g[l], this.vln2_g[l]);
      adamStep(this.ln2_b[l], this.dln2_b[l], this.mln2_b[l], this.vln2_b[l]);
      adamStep(this.wup[l], this.dwup[l], this.mwup[l], this.vwup[l]);
      adamStep(this.wdown[l], this.dwdown[l], this.mwdown[l], this.vwdown[l]);
    }
    adamStep(this.lnf_g, this.dlnf_g, this.mlnf_g, this.vlnf_g);
    adamStep(this.lnf_b, this.dlnf_b, this.mlnf_b, this.vlnf_b);
    adamStep(this.wlm, this.dwlm, this.mwlm, this.vwlm);
  }

  // ── Training forward ───────────────────────────────────
  forward(input, target) {
    const T = input.length;
    const { vocabSize, dModel, nLayers, nHeads, dHead, dFF } = this;

    // Embeddings
    const embeds = new Float32Array(T * dModel);
    for (let t = 0; t < T; t++) {
      const tok = input[t];
      const pos = Math.min(t, this.maxSeqLen - 1);
      const eBase = t * dModel;
      const teBase = tok * dModel;
      const peBase = pos * dModel;
      for (let d = 0; d < dModel; d++) {
        embeds[eBase + d] = this.wte[teBase + d] + this.wpe[peBase + d];
      }
    }

    let x = embeds;
    const layerCaches = [];

    for (let l = 0; l < nLayers; l++) {
      const residual1 = new Float32Array(x);

      // LN1
      const ln1 = layerNormForward(x, this.ln1_g[l], this.ln1_b[l]);

      // QKV
      const qkv = matMul(ln1.y, this.wqkv[l], T, dModel, 3 * dModel);

      // Multi-head causal attention
      const attnOut = new Float32Array(T * dModel);
      const attnWeights = new Float32Array(nHeads * T * T);

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

        // Scores
        const scores = matMulBT(Q, K, T, T, dHead);
        const scale = 1.0 / Math.sqrt(dHead);
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            if (j > i) scores[i * T + j] = -1e9;
            else scores[i * T + j] *= scale;
          }
        }

        const weights = softmaxRows(scores, T, T);
        for (let i = 0; i < T * T; i++) {
          attnWeights[h * T * T + i] = weights[i];
        }

        const outH = matMul(weights, V, T, T, dHead);
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < dHead; d++) {
            attnOut[t * dModel + h * dHead + d] = outH[t * dHead + d];
          }
        }
      }

      // Output projection
      const attnProj = matMul(attnOut, this.wo[l], T, dModel, dModel);

      // Residual
      for (let i = 0; i < x.length; i++) x[i] = residual1[i] + attnProj[i];

      const residual2 = new Float32Array(x);

      // LN2
      const ln2 = layerNormForward(x, this.ln2_g[l], this.ln2_b[l]);

      // FFN
      const ffnPre = matMul(ln2.y, this.wup[l], T, dModel, dFF);
      const ffnMid = gelu(ffnPre);
      const ffnOut = matMul(ffnMid, this.wdown[l], T, dFF, dModel);

      // Residual
      for (let i = 0; i < x.length; i++) x[i] = residual2[i] + ffnOut[i];

      layerCaches.push({
        x: residual1,
        ln1,
        qkv,
        attnWeights,
        attnOut,
        attnProj,
        residual2,
        ln2,
        ffnPre,
        ffnMid,
        ffnOut,
      });
    }

    // Final LN
    const lnF = layerNormForward(x, this.lnf_g, this.lnf_b);

    // LM head
    const logits = matMul(lnF.y, this.wlm, T, dModel, vocabSize);

    // Loss
    let loss = 0;
    for (let t = 0; t < T; t++) {
      const logitSlice = logits.subarray(t * vocabSize, (t + 1) * vocabSize);
      loss += crossEntropyLoss(logitSlice, target[t]);
    }
    loss /= T;

    return {
      loss,
      cache: {
        T,
        input,
        layers: layerCaches,
        finalX: x,
        lnF,
        logits,
      },
    };
  }

  // ── Backward ───────────────────────────────────────────
  backward(cache) {
    const { T, input, layers, finalX, lnF, logits } = cache;
    const { vocabSize, dModel, nLayers, nHeads, dHead, dFF } = this;

    // dLogits
    const dLogits = new Float32Array(T * vocabSize);
    for (let t = 0; t < T; t++) {
      const slice = crossEntropyBackward(
        logits.subarray(t * vocabSize, (t + 1) * vocabSize),
      );
      for (let v = 0; v < vocabSize; v++) {
        dLogits[t * vocabSize + v] = slice[v] / T;
      }
    }

    // LM head
    const dLnF_y = matMulBT(dLogits, this.wlm, T, dModel, vocabSize);
    const dWlm = matMulAT(lnF.y, dLogits, T, dModel, vocabSize);
    addInPlace(this.dwlm, dWlm);

    // Final LN
    const lnFGrad = layerNormBackward(
      dLnF_y,
      finalX,
      this.lnf_g,
      lnF.mean,
      lnF.rstd,
    );
    addInPlace(this.dlnf_g, lnFGrad.dgamma);
    addInPlace(this.dlnf_b, lnFGrad.dbeta);

    let dx = lnFGrad.dx;

    for (let l = nLayers - 1; l >= 0; l--) {
      const lc = layers[l];

      // Second residual: x_out = residual2 + ffnOut
      const dFfnOut = dx;
      const dResidual2 = new Float32Array(dx);

      // FFN backward
      const dFfnMid = matMulBT(dFfnOut, this.wdown[l], T, dFF, dModel);
      const dWdown = matMulAT(lc.ffnMid, dFfnOut, T, dFF, dModel);
      addInPlace(this.dwdown[l], dWdown);

      const dFfnPre = geluDeriv(lc.ffnPre, dFfnMid);

      const dLn2_y = matMulBT(dFfnPre, this.wup[l], T, dModel, dFF);
      const dWup = matMulAT(lc.ln2.y, dFfnPre, T, dModel, dFF);
      addInPlace(this.dwup[l], dWup);

      const ln2Grad = layerNormBackward(
        dLn2_y,
        lc.residual2,
        this.ln2_g[l],
        lc.ln2.mean,
        lc.ln2.rstd,
      );
      addInPlace(this.dln2_g[l], ln2Grad.dgamma);
      addInPlace(this.dln2_b[l], ln2Grad.dbeta);

      for (let i = 0; i < dx.length; i++) dResidual2[i] += ln2Grad.dx[i];

      // First residual: x_mid = residual1 + attnProj
      const dAttnProj = dResidual2;
      const dResidual1 = new Float32Array(dResidual2);

      const dAttnOut = matMulBT(dAttnProj, this.wo[l], T, dModel, dModel);
      const dWo = matMulAT(lc.attnOut, dAttnProj, T, dModel, dModel);
      addInPlace(this.dwo[l], dWo);

      // Multi-head attention backward
      const dQkv = new Float32Array(T * 3 * dModel);

      for (let h = 0; h < nHeads; h++) {
        const dAttnOut_h = new Float32Array(T * dHead);
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < dHead; d++) {
            dAttnOut_h[t * dHead + d] = dAttnOut[t * dModel + h * dHead + d];
          }
        }

        const V = new Float32Array(T * dHead);
        const weights = new Float32Array(T * T);
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < dHead; d++) {
            V[t * dHead + d] =
              lc.qkv[t * 3 * dModel + 2 * dModel + h * dHead + d];
          }
        }
        for (let i = 0; i < T * T; i++) {
          weights[i] = lc.attnWeights[h * T * T + i];
        }

        const dWeights = matMulBT(dAttnOut_h, V, T, T, dHead);
        const dV = matMulAT(weights, dAttnOut_h, T, T, dHead);

        // Softmax backward
        const dScores = new Float32Array(T * T);
        for (let i = 0; i < T; i++) {
          const wRow = weights.subarray(i * T, (i + 1) * T);
          const dwRow = dWeights.subarray(i * T, (i + 1) * T);
          let dot = 0;
          for (let j = 0; j < T; j++) dot += dwRow[j] * wRow[j];
          for (let j = 0; j < T; j++) {
            dScores[i * T + j] = wRow[j] * (dwRow[j] - dot);
          }
        }

        const scale = 1.0 / Math.sqrt(dHead);
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            if (j > i) dScores[i * T + j] = 0;
            else dScores[i * T + j] *= scale;
          }
        }

        const Q = new Float32Array(T * dHead);
        const K = new Float32Array(T * dHead);
        for (let t = 0; t < T; t++) {
          for (let d = 0; d < dHead; d++) {
            Q[t * dHead + d] = lc.qkv[t * 3 * dModel + h * dHead + d];
            K[t * dHead + d] = lc.qkv[t * 3 * dModel + dModel + h * dHead + d];
          }
        }

        const dQ = matMul(dScores, K, T, T, dHead);
        const dK = matMulAT(dScores, Q, T, T, dHead);

        for (let t = 0; t < T; t++) {
          for (let d = 0; d < dHead; d++) {
            dQkv[t * 3 * dModel + h * dHead + d] += dQ[t * dHead + d];
            dQkv[t * 3 * dModel + dModel + h * dHead + d] += dK[t * dHead + d];
            dQkv[t * 3 * dModel + 2 * dModel + h * dHead + d] +=
              dV[t * dHead + d];
          }
        }
      }

      const dLn1_y = matMulBT(dQkv, this.wqkv[l], T, dModel, 3 * dModel);
      const dWqkv = matMulAT(lc.ln1.y, dQkv, T, dModel, 3 * dModel);
      addInPlace(this.dwqkv[l], dWqkv);

      const ln1Grad = layerNormBackward(
        dLn1_y,
        lc.x,
        this.ln1_g[l],
        lc.ln1.mean,
        lc.ln1.rstd,
      );
      addInPlace(this.dln1_g[l], ln1Grad.dgamma);
      addInPlace(this.dln1_b[l], ln1Grad.dbeta);

      for (let i = 0; i < dx.length; i++) dResidual1[i] += ln1Grad.dx[i];
      dx = dResidual1;
    }

    // Embedding gradients
    for (let t = 0; t < T; t++) {
      const tok = input[t];
      const teBase = tok * dModel;
      const eBase = t * dModel;
      const peBase = t * dModel;
      for (let d = 0; d < dModel; d++) {
        this.dwte[teBase + d] += dx[eBase + d];
        this.dwpe[peBase + d] += dx[eBase + d];
      }
    }
  }

  // ── Training step ──────────────────────────────────────
  trainBatch(inputs, targets) {
    this.zeroGrad();
    let totalLoss = 0;
    const count = inputs.length;
    for (let b = 0; b < count; b++) {
      const { loss, cache } = this.forward(inputs[b], targets[b]);
      this.backward(cache);
      totalLoss += loss;
    }
    this.clipGrads();
    this.update();
    return totalLoss / count;
  }

  // ── Inference step with KV-cache ───────────────────────
  forwardStep(tokenId, pos, kvCaches) {
    const { dModel, nLayers, nHeads, dHead, dFF, vocabSize } = this;

    let x = new Float32Array(dModel);
    const teBase = tokenId * dModel;
    const peBase = pos * dModel;
    for (let d = 0; d < dModel; d++) {
      x[d] = this.wte[teBase + d] + this.wpe[peBase + d];
    }

    for (let l = 0; l < nLayers; l++) {
      const residual1 = new Float32Array(x);

      // LN1
      const ln1 = layerNormForward(x, this.ln1_g[l], this.ln1_b[l]);

      // QKV for single token
      const qkv = new Float32Array(3 * dModel);
      for (let j = 0; j < 3 * dModel; j++) {
        let sum = 0;
        for (let d = 0; d < dModel; d++) {
          sum += ln1.y[d] * this.wqkv[l][d * 3 * dModel + j];
        }
        qkv[j] = sum;
      }

      const q = new Float32Array(dModel);
      const k = new Float32Array(dModel);
      const v = new Float32Array(dModel);
      for (let d = 0; d < dModel; d++) {
        q[d] = qkv[d];
        k[d] = qkv[dModel + d];
        v[d] = qkv[2 * dModel + d];
      }

      // Update cache
      const cache = kvCaches[l];
      const kBase = pos * dModel;
      for (let d = 0; d < dModel; d++) {
        cache.K[kBase + d] = k[d];
        cache.V[kBase + d] = v[d];
      }

      // Multi-head causal attention
      const attnOut = new Float32Array(dModel);
      const scale = 1.0 / Math.sqrt(dHead);

      for (let h = 0; h < nHeads; h++) {
        const scores = new Float32Array(pos + 1);
        for (let p = 0; p <= pos; p++) {
          let dot = 0;
          for (let d = 0; d < dHead; d++) {
            dot += q[h * dHead + d] * cache.K[p * dModel + h * dHead + d];
          }
          scores[p] = dot * scale;
        }

        const weights = (() => {
          let max = -Infinity;
          for (let i = 0; i < scores.length; i++)
            max = Math.max(max, scores[i]);
          let sum = 0;
          const out = new Float32Array(scores.length);
          for (let i = 0; i < scores.length; i++) {
            out[i] = Math.exp(scores[i] - max);
            sum += out[i];
          }
          for (let i = 0; i < scores.length; i++) out[i] /= sum;
          return out;
        })();

        for (let d = 0; d < dHead; d++) {
          let sum = 0;
          for (let p = 0; p <= pos; p++) {
            sum += weights[p] * cache.V[p * dModel + h * dHead + d];
          }
          attnOut[h * dHead + d] = sum;
        }
      }

      // Output projection
      const attnProj = new Float32Array(dModel);
      for (let j = 0; j < dModel; j++) {
        let sum = 0;
        for (let d = 0; d < dModel; d++) {
          sum += attnOut[d] * this.wo[l][d * dModel + j];
        }
        attnProj[j] = sum;
      }

      for (let d = 0; d < dModel; d++) x[d] = residual1[d] + attnProj[d];

      const residual2 = new Float32Array(x);

      // LN2
      const ln2 = layerNormForward(x, this.ln2_g[l], this.ln2_b[l]);

      // FFN
      const ffnPre = new Float32Array(dFF);
      for (let j = 0; j < dFF; j++) {
        let sum = 0;
        for (let d = 0; d < dModel; d++) {
          sum += ln2.y[d] * this.wup[l][d * dFF + j];
        }
        ffnPre[j] = sum;
      }

      const ffnMid = gelu(ffnPre);

      const ffnOut = new Float32Array(dModel);
      for (let j = 0; j < dModel; j++) {
        let sum = 0;
        for (let d = 0; d < dFF; d++) {
          sum += ffnMid[d] * this.wdown[l][d * dModel + j];
        }
        ffnOut[j] = sum;
      }

      for (let d = 0; d < dModel; d++) x[d] = residual2[d] + ffnOut[d];
    }

    // Final LN
    const lnF = layerNormForward(x, this.lnf_g, this.lnf_b);

    // LM head logits
    const logits = new Float32Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      let sum = 0;
      for (let d = 0; d < dModel; d++) {
        sum += lnF.y[d] * this.wlm[d * vocabSize + v];
      }
      logits[v] = sum;
    }

    return logits;
  }

  // ── Generation with KV-cache ───────────────────────────
  generate(seedTokens, maxLen = 100, temperature = 0.8) {
    const kvCaches = [];
    for (let l = 0; l < this.nLayers; l++) {
      kvCaches.push({
        K: new Float32Array(this.maxSeqLen * this.dModel),
        V: new Float32Array(this.maxSeqLen * this.dModel),
      });
    }

    const out = [];
    let pos = 0;

    // Burn-in seed
    for (const tok of seedTokens) {
      this.forwardStep(tok, pos, kvCaches);
      pos++;
    }

    let last = seedTokens.length ? seedTokens[seedTokens.length - 1] : 0;

    for (let i = 0; i < maxLen; i++) {
      if (pos >= this.maxSeqLen) break;
      const logits = this.forwardStep(last, pos, kvCaches);
      last = sample(logits, temperature);
      if (last === 0) break;
      out.push(last);
      pos++;
    }

    return out;
  }

  // ── Weight I/O ─────────────────────────────────────────
  _allWeights() {
    const all = [
      this.wte,
      this.wpe,
      ...this.ln1_g,
      ...this.ln1_b,
      ...this.wqkv,
      ...this.wo,
      ...this.ln2_g,
      ...this.ln2_b,
      ...this.wup,
      ...this.wdown,
      this.lnf_g,
      this.lnf_b,
      this.wlm,
    ];
    let total = 0;
    for (const a of all) total += a.length;
    const buf = new Float32Array(total);
    let off = 0;
    for (const a of all) {
      buf.set(a, off);
      off += a.length;
    }
    return Buffer.from(buf.buffer);
  }

  _setAllWeights(buf) {
    const arr = new Float32Array(
      buf.buffer,
      buf.byteOffset,
      buf.byteLength / 4,
    );
    const all = [
      this.wte,
      this.wpe,
      ...this.ln1_g,
      ...this.ln1_b,
      ...this.wqkv,
      ...this.wo,
      ...this.ln2_g,
      ...this.ln2_b,
      ...this.wup,
      ...this.wdown,
      this.lnf_g,
      this.lnf_b,
      this.wlm,
    ];
    let off = 0;
    for (const a of all) {
      a.set(arr.subarray(off, off + a.length));
      off += a.length;
    }
  }

  saveWeights(path) {
    fs.writeFileSync(path, this._allWeights());
  }

  loadWeights(path) {
    if (!fs.existsSync(path)) return;
    const buf = fs.readFileSync(path);
    this._setAllWeights(buf);
  }

  clone() {
    const m = new Transformer(
      this.lr,
      this.vocabSize,
      this.dModel,
      this.nLayers,
      this.nHeads,
      this.dFF,
      this.maxSeqLen,
      this.beta1,
      this.beta2,
      this.eps,
    );
    m.wte.set(this.wte);
    m.wpe.set(this.wpe);
    for (let l = 0; l < this.nLayers; l++) {
      m.ln1_g[l].set(this.ln1_g[l]);
      m.ln1_b[l].set(this.ln1_b[l]);
      m.wqkv[l].set(this.wqkv[l]);
      m.wo[l].set(this.wo[l]);
      m.ln2_g[l].set(this.ln2_g[l]);
      m.ln2_b[l].set(this.ln2_b[l]);
      m.wup[l].set(this.wup[l]);
      m.wdown[l].set(this.wdown[l]);
    }
    m.lnf_g.set(this.lnf_g);
    m.lnf_b.set(this.lnf_b);
    m.wlm.set(this.wlm);
    return m;
  }
}

module.exports = Transformer;
