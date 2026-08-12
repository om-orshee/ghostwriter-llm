// server.js
const http = require('http');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');

// ── Load .env if present (zero dependencies) ───────────
function loadEnv(path = './.env') {
  if (!fs.existsSync(path)) return;
  const text = fs.readFileSync(path, 'utf8');
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq === -1) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim();
    if (key && process.env[key] === undefined) {
      process.env[key] = val;
    }
  }
}
loadEnv();

const Transformer = require('./transformer');
const { BPETokenizer } = require('./tokenizer');
const { Ghost } = require('./ghost');
const { ParallelTrainer } = require('./trainer');

const TOKENIZER_PATH = './training/tokenizer.json';
const BASE_WEIGHTS_PATH = './weights.bin';
const GHOST_DIR = './ghosts';
const CHECKPOINT_DIR = './checkpoints';
const CHECKPOINT_STATE = `${CHECKPOINT_DIR}/base_training_state.json`;
const CHECKPOINT_WEIGHTS = `${CHECKPOINT_DIR}/base_weights_checkpoint.bin`;

// ── Env toggles ────────────────────────────────────────
const TEST_MODE = process.env.TEST_MODE !== 'false';
const RESUME = process.env.RESUME === 'true';

// ── Training state (shared across endpoints) ───────────
const trainingStatus = {
  isTraining: false,
  shouldPause: false,
  currentEpoch: 0,
  currentBatch: 0,
  totalBatches: 0,
  currentLoss: 0,
  etaSeconds: 0,
};

// ── Init tokenizer ─────────────────────────────────────
const tokenizer = new BPETokenizer(512);

if (fs.existsSync(TOKENIZER_PATH)) {
  tokenizer.load(TOKENIZER_PATH);
  console.log(
    `Tokenizer loaded: ${tokenizer.vocabSize} tokens, ${tokenizer.merges.length} merges`,
  );
} else {
  console.log('No tokenizer found. Train base model with text to build it.');
}

// ── Init base model ────────────────────────────────────
const baseModel = TEST_MODE
  ? new Transformer(0.001, 512, 128, 4, 4, 512, 128) // ~1.1M params
  : new Transformer(0.001, 512, 256, 6, 8, 768, 256); // ~4.3M params

console.log(
  `Transformer initialized: ${(baseModel.paramCount() / 1e6).toFixed(2)}M params`,
);
console.log(
  `Mode: ${TEST_MODE ? 'TEST (small, fast)' : 'FULL (large, data-hungry)'}`,
);
console.log(`Workers: ${process.env.WORKERS || os.cpus().length}`);
console.log(`Resume: ${RESUME}`);
console.log('');

try {
  baseModel.loadWeights(BASE_WEIGHTS_PATH);
  console.log('Base model weights loaded.');
} catch (e) {
  console.log('No base weights found. Run POST /train with text first.');
}

const baseTrainer = new ParallelTrainer(
  baseModel,
  parseInt(process.env.WORKERS || os.cpus().length, 10),
);

// ── Checkpoint helpers ─────────────────────────────────
function saveCheckpoint(state) {
  if (!fs.existsSync(CHECKPOINT_DIR)) fs.mkdirSync(CHECKPOINT_DIR);
  fs.writeFileSync(CHECKPOINT_STATE, JSON.stringify(state, null, 2));
  baseModel.saveWeights(CHECKPOINT_WEIGHTS);
}

function loadCheckpoint() {
  if (!fs.existsSync(CHECKPOINT_STATE)) return null;
  if (!fs.existsSync(CHECKPOINT_WEIGHTS)) return null;
  try {
    const state = JSON.parse(fs.readFileSync(CHECKPOINT_STATE, 'utf8'));
    baseModel.loadWeights(CHECKPOINT_WEIGHTS);
    return state;
  } catch (e) {
    console.log('[CHECKPOINT] Corrupted checkpoint, starting fresh.');
    deleteCheckpoint();
    return null;
  }
}

function deleteCheckpoint() {
  try {
    fs.unlinkSync(CHECKPOINT_STATE);
  } catch (e) {}
  try {
    fs.unlinkSync(CHECKPOINT_WEIGHTS);
  } catch (e) {}
}

// ── Init ghosts ────────────────────────────────────────
const ghosts = new Map();

if (!fs.existsSync(GHOST_DIR)) fs.mkdirSync(GHOST_DIR);

try {
  const files = fs.readdirSync(GHOST_DIR).filter((f) => f.endsWith('.json'));
  for (const file of files) {
    const meta = JSON.parse(fs.readFileSync(`${GHOST_DIR}/${file}`, 'utf8'));
    const ghostModel = baseModel.clone();
    const g = Ghost.fromJSON(meta, ghostModel, tokenizer);
    const weightPath = `${GHOST_DIR}/${g.name}_weights.bin`;
    if (fs.existsSync(weightPath)) {
      g.loadWeights(weightPath);
      console.log(`Loaded personality weights for ${g.name}`);
    }
    ghosts.set(g.name, g);
  }
} catch (e) {
  console.log('No ghosts found, creating defaults...');
}

if (ghosts.size === 0) {
  const veraModel = baseModel.clone();
  const jaxModel = baseModel.clone();

  const vera = new Ghost(
    'Vera',
    'You are Vera, a gentle clockmaker from Prague. You speak slowly and thoughtfully.',
    veraModel,
    tokenizer,
  );
  const jax = new Ghost(
    'Jax',
    'You are Jax, a street racer from Miami. You speak fast and use slang.',
    jaxModel,
    tokenizer,
  );

  ghosts.set('Vera', vera);
  ghosts.set('Jax', jax);
}

function saveGhosts() {
  for (const [name, g] of ghosts) {
    fs.writeFileSync(
      `${GHOST_DIR}/${name}.json`,
      JSON.stringify(g.toJSON(), null, 2),
    );
    g.saveWeights(`${GHOST_DIR}/${name}_weights.bin`);
  }
}

function json(res, status, data) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.end(JSON.stringify(data));
}

// ── HTTP Server ────────────────────────────────────────
const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(200, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    });
    res.end();
    return;
  }

  const parsed = new URL(req.url, `http://${req.headers.host}`);
  let body = '';

  req.on('data', (chunk) => (body += chunk));
  req.on('end', async () => {
    const data = body ? JSON.parse(body) : {};

    // GET /training-status
    if (parsed.pathname === '/training-status' && req.method === 'GET') {
      return json(res, 200, {
        isTraining: trainingStatus.isTraining,
        shouldPause: trainingStatus.shouldPause,
        currentEpoch: trainingStatus.currentEpoch,
        currentBatch: trainingStatus.currentBatch,
        totalBatches: trainingStatus.totalBatches,
        currentLoss: trainingStatus.currentLoss,
        etaSeconds: trainingStatus.etaSeconds,
      });
    }

    // POST /pause-training
    if (parsed.pathname === '/pause-training' && req.method === 'POST') {
      if (!trainingStatus.isTraining) {
        return json(res, 409, { error: 'No training in progress' });
      }
      trainingStatus.shouldPause = true;
      return json(res, 200, {
        status: 'pause_requested',
        message: 'Training will pause after the current batch completes.',
      });
    }

    // POST /speak
    if (parsed.pathname === '/speak' && req.method === 'POST') {
      const g = ghosts.get(data.ghost);
      if (!g) return json(res, 404, { error: 'Ghost not found' });
      const reply = g.speak(
        data.message,
        data.temperature ?? 0.7,
        data.learn !== false,
      );
      saveGhosts();
      return json(res, 200, { ghost: data.ghost, reply });
    }

    // GET /ghosts
    if (parsed.pathname === '/ghosts' && req.method === 'GET') {
      return json(res, 200, [...ghosts.keys()]);
    }

    // POST /ghosts
    if (parsed.pathname === '/ghosts' && req.method === 'POST') {
      if (ghosts.has(data.name))
        return json(res, 409, { error: 'Ghost already exists' });
      const ghostModel = baseModel.clone();
      const g = new Ghost(data.name, data.backstory, ghostModel, tokenizer);
      ghosts.set(data.name, g);
      saveGhosts();
      return json(res, 200, {
        name: data.name,
        note: 'Created with base weights cloned. Train with /train-ghost to specialize.',
      });
    }

    // GET /history?ghost=NAME
    if (parsed.pathname === '/history' && req.method === 'GET') {
      const g = ghosts.get(parsed.searchParams.get('ghost'));
      if (!g) return json(res, 404, { error: 'Ghost not found' });
      return json(res, 200, g.history);
    }

    // POST /train — train the BASE model (with resume support)
    if (parsed.pathname === '/train' && req.method === 'POST') {
      if (trainingStatus.isTraining) {
        return json(res, 409, {
          error:
            'Training already in progress. Use /pause-training first or wait.',
        });
      }

      let text = data.text || '';

      if (data.file) {
        try {
          text = fs.readFileSync(data.file, 'utf8');
          console.log(
            `[TRAIN] Loaded file: ${data.file} (${text.length.toLocaleString()} chars)`,
          );
        } catch (e) {
          return json(res, 400, { error: 'Cannot read file: ' + data.file });
        }
      }

      if (!text) return json(res, 400, { error: 'Provide text or file' });

      // Train tokenizer on first run
      if (tokenizer.merges.length === 0) {
        console.log('[TRAIN] Training tokenizer on provided text...');
        tokenizer.train(text, tokenizer.vocabSize - 256);
        tokenizer.save(TOKENIZER_PATH);
        console.log(`[TRAIN] Tokenizer saved to ${TOKENIZER_PATH}`);
      }

      const epochs = data.epochs || (TEST_MODE ? 20 : 5);
      const batchSize = data.batchSize || 16;
      const stride = data.stride || (TEST_MODE ? 32 : 64);
      const seqLen = baseModel.maxSeqLen;

      const tokens = tokenizer.encode(text);
      console.log(
        `[TRAIN] Tokenized to ${tokens.length.toLocaleString()} tokens`,
      );

      const sequences = [];
      const effectiveStride = stride > 0 && stride < seqLen ? stride : seqLen;
      for (let i = 0; i < tokens.length - seqLen; i += effectiveStride) {
        sequences.push({
          inp: new Int32Array(tokens.slice(i, i + seqLen)),
          tgt: new Int32Array(tokens.slice(i + 1, i + seqLen + 1)),
        });
      }

      if (sequences.length === 0) {
        return json(res, 400, { error: 'Text too short after tokenization' });
      }

      const batchesPerEpoch = Math.ceil(sequences.length / batchSize);
      const totalBatches = batchesPerEpoch * epochs;

      // ── Resume logic ─────────────────────────────────
      let checkpoint = null;
      let nextGlobalBatch = 0;
      let totalLoss = 0;
      let steps = 0;
      let startTime = Date.now();
      let resuming = false;

      if (RESUME) {
        checkpoint = loadCheckpoint();
      }

      if (checkpoint && checkpoint.status === 'paused') {
        // Validate checkpoint matches current request
        const configMatch =
          checkpoint.config &&
          checkpoint.config.file === (data.file || '') &&
          checkpoint.config.textLength === text.length &&
          checkpoint.config.batchSize === batchSize &&
          checkpoint.config.stride === effectiveStride &&
          checkpoint.config.seqLen === seqLen;

        if (
          configMatch &&
          checkpoint.nextGlobalBatch < checkpoint.config.totalBatches
        ) {
          console.log(
            `[TRAIN] Resuming from checkpoint: epoch ${checkpoint.progress.epoch + 1}, batch ${checkpoint.progress.batch + 1}`,
          );
          nextGlobalBatch = checkpoint.nextGlobalBatch;
          totalLoss = checkpoint.totalLoss || 0;
          steps = checkpoint.steps || 0;
          startTime = checkpoint.startTime || Date.now();
          resuming = true;
        } else {
          console.log(
            '[TRAIN] Checkpoint mismatch or completed. Starting fresh.',
          );
          deleteCheckpoint();
        }
      } else if (!RESUME && checkpoint) {
        console.log('[TRAIN] RESUME=false, deleting old checkpoint.');
        deleteCheckpoint();
      }

      if (!resuming) {
        deleteCheckpoint();
      }

      console.log(
        `[TRAIN] ${epochs} epochs | ${sequences.length.toLocaleString()} sequences | batch=${batchSize} | stride=${effectiveStride} | ${totalBatches} total batches`,
      );
      if (resuming) {
        console.log(
          `[TRAIN] Resuming at global batch ${nextGlobalBatch}/${totalBatches}`,
        );
      }
      console.log(`[TRAIN] Starting training... (logging every batch)\n`);

      trainingStatus.isTraining = true;
      trainingStatus.shouldPause = false;
      trainingStatus.totalBatches = totalBatches;

      let batchStart = Date.now();
      let paused = false;

      try {
        for (let e = 0; e < epochs; e++) {
          let epochLoss = 0;

          for (let b = 0; b < batchesPerEpoch; b++) {
            const globalBatch = e * batchesPerEpoch + b;

            // Skip already-completed batches when resuming
            if (globalBatch < nextGlobalBatch) continue;

            // Check pause request
            if (trainingStatus.shouldPause) {
              console.log(
                `[TRAIN] Pause requested at batch ${globalBatch}. Saving checkpoint...`,
              );
              saveCheckpoint({
                status: 'paused',
                config: {
                  file: data.file || '',
                  textLength: text.length,
                  epochs,
                  batchSize,
                  stride: effectiveStride,
                  seqLen,
                  totalBatches,
                },
                progress: { epoch: e, batch: b },
                nextGlobalBatch: globalBatch,
                totalLoss,
                steps,
                startTime,
              });
              paused = true;
              break;
            }

            const i = b * batchSize;
            const batch = sequences.slice(i, i + batchSize);
            const loss = await baseTrainer.trainBatch(
              batch.map((s) => s.inp),
              batch.map((s) => s.tgt),
            );

            totalLoss += loss * batch.length;
            steps += batch.length;
            epochLoss += loss * batch.length;

            // Update live status
            trainingStatus.currentEpoch = e + 1;
            trainingStatus.currentBatch = globalBatch + 1;
            trainingStatus.currentLoss = loss;
            const now = Date.now();
            const batchMs = now - batchStart;
            batchStart = now;
            const elapsedSec = (now - startTime) / 1000;
            const remainingBatches = totalBatches - globalBatch - 1;
            trainingStatus.etaSeconds = Math.round(
              (remainingBatches * batchMs) / 1000,
            );

            const elapsedStr = elapsedSec.toFixed(1);
            const etaStr = trainingStatus.etaSeconds.toFixed(0);

            console.log(
              `[TRAIN] ${globalBatch + 1}/${totalBatches} | Epoch ${e + 1}/${epochs} | ` +
                `Loss: ${loss.toFixed(4)} | Batch: ${batchMs}ms | Elapsed: ${elapsedStr}s | ETA: ${etaStr}s`,
            );
          }

          if (paused) break;

          console.log(
            `\n[TRAIN] Epoch ${e + 1}/${epochs} complete | Avg loss: ${(epochLoss / sequences.length).toFixed(4)}\n`,
          );
        }
      } finally {
        trainingStatus.isTraining = false;
        trainingStatus.shouldPause = false;
      }

      if (paused) {
        return json(res, 200, {
          status: 'paused',
          message:
            'Training paused. Call POST /train to resume (with RESUME=true).',
          checkpoint: CHECKPOINT_STATE,
          progress: {
            nextGlobalBatch,
            totalBatches,
            totalLoss: totalLoss / (steps || 1),
          },
        });
      }

      // Training completed
      baseModel.saveWeights(BASE_WEIGHTS_PATH);
      const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(
        `[TRAIN] Done in ${totalTime}s. Saved base weights to ${BASE_WEIGHTS_PATH}`,
      );

      deleteCheckpoint();

      // Sync into ghosts
      for (const [name, g] of ghosts) {
        g.model.wte.set(baseModel.wte);
        g.model.wpe.set(baseModel.wpe);
        for (let l = 0; l < baseModel.nLayers; l++) {
          g.model.ln1_g[l].set(baseModel.ln1_g[l]);
          g.model.ln1_b[l].set(baseModel.ln1_b[l]);
          g.model.wqkv[l].set(baseModel.wqkv[l]);
          g.model.wo[l].set(baseModel.wo[l]);
          g.model.ln2_g[l].set(baseModel.ln2_g[l]);
          g.model.ln2_b[l].set(baseModel.ln2_b[l]);
          g.model.wup[l].set(baseModel.wup[l]);
          g.model.wdown[l].set(baseModel.wdown[l]);
        }
        g.model.lnf_g.set(baseModel.lnf_g);
        g.model.lnf_b.set(baseModel.lnf_b);
        g.model.wlm.set(baseModel.wlm);
        console.log(`[TRAIN] Synced updated base weights into ${name}`);
      }

      return json(res, 200, {
        status: 'completed',
        steps,
        avgLoss: totalLoss / (steps || 1),
        source: data.file || 'inline text',
        chars: text.length,
        tokens: tokens.length,
        batches: totalBatches,
        timeSeconds: parseFloat(totalTime),
      });
    }

    // POST /train-ghost — train a PERSONALITY
    if (parsed.pathname === '/train-ghost' && req.method === 'POST') {
      const g = ghosts.get(data.ghost);
      if (!g) return json(res, 404, { error: 'Ghost not found' });

      let text = data.text || '';

      if (data.file) {
        try {
          text = fs.readFileSync(data.file, 'utf8');
          console.log(
            `[TRAIN-GHOST] Loaded file for ${g.name}: ${data.file} (${text.length.toLocaleString()} chars)`,
          );
        } catch (e) {
          return json(res, 400, { error: 'Cannot read file: ' + data.file });
        }
      }

      if (!text) return json(res, 400, { error: 'Provide text or file' });

      const epochs = data.epochs || 10;
      const batchSize = data.batchSize || 8;
      console.log(
        `[TRAIN-GHOST] Training ${g.name}: ${epochs} epochs, batch=${batchSize}`,
      );

      g.train(text, epochs, batchSize);
      g.saveWeights(`${GHOST_DIR}/${g.name}_weights.bin`);

      console.log(`[TRAIN-GHOST] ${g.name} weights saved.`);

      return json(res, 200, {
        ghost: g.name,
        epochs,
        chars: text.length,
        status: 'Personality weights updated',
      });
    }

    // GET /
    if (parsed.pathname === '/' && req.method === 'GET') {
      return json(res, 200, {
        info: 'Ghostwriter — Transformer + BPE + Personality Weights',
        mode: TEST_MODE
          ? 'TEST (1.1M params, fast)'
          : 'FULL (4.3M params, data-hungry)',
        resume: RESUME,
        params: {
          vocabSize: tokenizer.vocabSize,
          dModel: baseModel.dModel,
          nLayers: baseModel.nLayers,
          nHeads: baseModel.nHeads,
          dFF: baseModel.dFF,
          maxSeqLen: baseModel.maxSeqLen,
          totalParams: `${(baseModel.paramCount() / 1e6).toFixed(2)}M`,
        },
        endpoints: {
          'POST /train':
            'Train base model (also builds tokenizer on first run)',
          'POST /pause-training': 'Request graceful pause of running training',
          'GET /training-status': 'Live training progress',
          'POST /train-ghost': 'Train personality weights for a specific ghost',
          'POST /speak': 'Talk to a ghost (uses personal model + history)',
          'GET /ghosts': 'List ghosts',
          'POST /ghosts': 'Create a ghost',
          'GET /history?ghost=NAME': 'View short-term text history',
        },
        trainingTips: {
          stride: '64 = max speed, 32 = medium, 8 = slow but thorough',
          batchSize: '16 recommended for base, 8 for ghosts',
          epochs: TEST_MODE
            ? '20+ for test mode (small data OK)'
            : '50+ for full mode (needs MBs of text)',
          data: 'Synthetic data is fine for testing. For real quality, feed books, Wikipedia, or code.',
        },
      });
    }

    json(res, 404, { error: 'Not found' });
  });
});

server.listen(3000, () =>
  console.log('Ghostwriter running on http://localhost:3000'),
);

process.on('SIGINT', () => {
  console.log('\nSIGINT received.');
  if (trainingStatus.isTraining) {
    console.log('Training is running. Requesting pause to save checkpoint...');
    trainingStatus.shouldPause = true;
    // Give the loop one batch worth of time to save checkpoint, then force exit
    setTimeout(() => {
      console.log('Forcing exit.');
      baseTrainer.terminate();
      saveGhosts();
      process.exit(0);
    }, 30000); // 30s grace period
  } else {
    baseTrainer.terminate();
    saveGhosts();
    process.exit(0);
  }
});
