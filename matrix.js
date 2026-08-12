//matrix.js
// matrix.js — Tensor primitives for a small transformer in pure JS
// All ops use Float32Array, row-major layout.

function zeros(n) {
  return new Float32Array(n);
}

function randn(n, scale = 1.0) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale;
  return a;
}

// Xavier / He style init scaled by fan-in
function randnInit(n, fanIn) {
  const scale = Math.sqrt(2.0 / fanIn);
  return randn(n, scale);
}

// C = A @ B   (A: M×K, B: K×N, C: M×N)
function matMul(A, B, M, K, N) {
  const C = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    const aRow = i * K;
    const cRow = i * N;
    for (let k = 0; k < K; k++) {
      const a = A[aRow + k];
      const bRow = k * N;
      for (let j = 0; j < N; j++) {
        C[cRow + j] += a * B[bRow + j];
      }
    }
  }
  return C;
}

// C = A^T @ B   (A: M×K stored row-major, B: M×N, C: K×N)
function matMulAT(A, B, M, K, N) {
  const C = new Float32Array(K * N);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < K; k++) {
      const a = A[m * K + k];
      const cBase = k * N;
      const bBase = m * N;
      for (let n = 0; n < N; n++) {
        C[cBase + n] += a * B[bBase + n];
      }
    }
  }
  return C;
}

// C = A @ B^T   (A: M×N, B: K×N stored row-major, C: M×K)
function matMulBT(A, B, M, K, N) {
  const C = new Float32Array(M * K);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < K; k++) {
      let sum = 0;
      const aBase = m * N;
      const bBase = k * N;
      for (let n = 0; n < N; n++) {
        sum += A[aBase + n] * B[bBase + n];
      }
      C[m * K + k] = sum;
    }
  }
  return C;
}

function add(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function addInPlace(a, b) {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

function scaleInPlace(a, s) {
  for (let i = 0; i < a.length; i++) a[i] *= s;
}

// Row-wise softmax: x is (rows × cols) flat array
function softmaxRows(x, rows, cols) {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    let max = -Infinity;
    const base = i * cols;
    for (let j = 0; j < cols; j++) max = Math.max(max, x[base + j]);
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      const e = Math.exp(x[base + j] - max);
      out[base + j] = e;
      sum += e;
    }
    for (let j = 0; j < cols; j++) out[base + j] /= sum;
  }
  return out;
}

// Vector softmax (for single logits row)
function softmax(x) {
  let max = -Infinity;
  for (let i = 0; i < x.length; i++) max = Math.max(max, x[i]);
  let sum = 0;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const e = Math.exp(x[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < x.length; i++) out[i] /= sum;
  return out;
}

// LayerNorm forward. x is (N × D). gamma/beta are length D.
function layerNormForward(x, gamma, beta, eps = 1e-5) {
  const N = x.length / gamma.length;
  const D = gamma.length;
  const y = new Float32Array(x.length);
  const mean = new Float32Array(N);
  const rstd = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    const base = i * D;
    let sum = 0;
    for (let j = 0; j < D; j++) sum += x[base + j];
    const m = sum / D;
    mean[i] = m;

    let varSum = 0;
    for (let j = 0; j < D; j++) {
      const d = x[base + j] - m;
      varSum += d * d;
    }
    const rs = 1.0 / Math.sqrt(varSum / D + eps);
    rstd[i] = rs;

    for (let j = 0; j < D; j++) {
      const norm = (x[base + j] - m) * rs;
      y[base + j] = norm * gamma[j] + beta[j];
    }
  }
  return { y, mean, rstd };
}

// LayerNorm backward. Returns { dx, dgamma, dbeta }
function layerNormBackward(dy, x, gamma, mean, rstd) {
  const N = mean.length;
  const D = gamma.length;
  const dx = new Float32Array(x.length);
  const dgamma = new Float32Array(D);
  const dbeta = new Float32Array(D);

  for (let i = 0; i < N; i++) {
    const base = i * D;
    const m = mean[i];
    const rs = rstd[i];

    let dvar = 0;
    let dmean = 0;

    for (let j = 0; j < D; j++) {
      const norm = (x[base + j] - m) * rs;
      dbeta[j] += dy[base + j];
      dgamma[j] += dy[base + j] * norm;

      const dnorm = dy[base + j] * gamma[j];
      dx[base + j] = dnorm * rs;
      dvar += dnorm * (x[base + j] - m);
      dmean += dnorm;
    }

    dvar *= -0.5 * rs * rs * rs;
    dmean *= -rs;

    for (let j = 0; j < D; j++) {
      dx[base + j] += (dvar * 2 * (x[base + j] - m)) / D + dmean / D;
    }
  }
  return { dx, dgamma, dbeta };
}

// GELU forward (tanh approximation)
const GELU_C = Math.sqrt(2 / Math.PI); // ≈ 0.7978845608
const GELU_D = 0.044715;

function gelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const x3 = x[i] * x[i] * x[i];
    const t = Math.tanh(GELU_C * (x[i] + GELU_D * x3));
    out[i] = 0.5 * x[i] * (1 + t);
  }
  return out;
}

// GELU backward: given pre-activation x and dy (dL/dy), returns dL/dx
function geluDeriv(x, dy) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const x2 = x[i] * x[i];
    const x3 = x2 * x[i];
    const u = GELU_C * (x[i] + GELU_D * x3);
    const t = Math.tanh(u);
    const dt = (1 - t * t) * GELU_C * (1 + 3 * GELU_D * x2);
    out[i] = dy[i] * (0.5 * (1 + t) + 0.5 * x[i] * dt);
  }
  return out;
}

// Cross-entropy loss for a single logits vector
function crossEntropyLoss(logits, target) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  return Math.log(sum) + max - logits[target];
}

// Softmax + cross-entropy backward: returns dL/dlogits
function crossEntropyBackward(logits, target) {
  const dy = softmax(logits);
  dy[target] -= 1.0;
  return dy;
}

// Categorical sampling from logits with temperature
function sample(logits, temperature) {
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  let max = -Infinity;
  for (let i = 0; i < scaled.length; i++) max = Math.max(max, scaled[i]);
  let sum = 0;
  const probs = new Float32Array(scaled.length);
  for (let i = 0; i < scaled.length; i++) {
    probs[i] = Math.exp(scaled[i] - max);
    sum += probs[i];
  }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;

  let r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

module.exports = {
  zeros,
  randn,
  randnInit,
  matMul,
  matMulAT,
  matMulBT,
  add,
  addInPlace,
  scaleInPlace,
  softmax,
  softmaxRows,
  layerNormForward,
  layerNormBackward,
  gelu,
  geluDeriv,
  crossEntropyLoss,
  crossEntropyBackward,
  sample,
};
