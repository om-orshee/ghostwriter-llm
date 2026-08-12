// server.js
const http = require('http');
const fs = require('fs');
const Transformer = require('./transformer');
const { BPETokenizer } = require('./tokenizer');
const { Ghost } = require('./ghost');
const { ParallelTrainer } = require('./trainer');

const TOKENIZER_PATH = './tokenizer.json';
const BASE_WEIGHTS_PATH = './weights.bin';
const GHOST_DIR = './ghosts';

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
const baseModel = new Transformer(
  0.001, // lr
  tokenizer.vocabSize,
  256, // dModel
  6, // nLayers
  8, // nHeads
  768, // dFF
  256, // maxSeqLen
);

console.log(
  `Transformer initialized: ${(baseModel.paramCount() / 1e6).toFixed(2)}M params`,
);
console.log(
  'WARNING: Pure-JS transformer training is slow. Expect ~10-60s per batch on CPU.\n',
);

try {
  baseModel.loadWeights(BASE_WEIGHTS_PATH);
  console.log('Base model weights loaded.');
} catch (e) {
  console.log('No base weights found. Run POST /train with text first.');
}

// ── Init parallel trainer ─────────────────────────────
// add 2nd argument to limit workers, e.g. new ParallelTrainer(baseModel, 16)
// else it defaults to os.cpus().length
const baseTrainer = new ParallelTrainer(baseModel);

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

    // POST /train — train the BASE model
    if (parsed.pathname === '/train' && req.method === 'POST') {
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

      if (tokenizer.merges.length === 0) {
        console.log('[TRAIN] Training tokenizer on provided text...');
        tokenizer.train(text, tokenizer.vocabSize - 256);
        tokenizer.save(TOKENIZER_PATH);
        console.log(`[TRAIN] Tokenizer saved to ${TOKENIZER_PATH}`);
      }

      const epochs = data.epochs || 5;
      const batchSize = data.batchSize || 16;
      const stride = data.stride || 64;
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

      console.log(
        `[TRAIN] ${epochs} epochs | ${sequences.length.toLocaleString()} sequences | batch=${batchSize} | stride=${effectiveStride} | ${totalBatches} total batches`,
      );
      console.log(`[TRAIN] Starting training... (logging every batch)\n`);

      let totalLoss = 0;
      let steps = 0;
      const startTime = Date.now();
      let batchStart = startTime;

      for (let e = 0; e < epochs; e++) {
        let epochLoss = 0;

        for (let i = 0; i < sequences.length; i += batchSize) {
          const batch = sequences.slice(i, i + batchSize);
          const loss = await baseTrainer.trainBatch(
            batch.map((s) => s.inp),
            batch.map((s) => s.tgt),
          );

          totalLoss += loss * batch.length;
          steps += batch.length;
          epochLoss += loss * batch.length;

          const globalBatch =
            e * batchesPerEpoch + Math.floor(i / batchSize) + 1;
          const now = Date.now();
          const batchMs = now - batchStart;
          batchStart = now;
          const elapsedSec = ((now - startTime) / 1000).toFixed(1);
          const etaSec = (
            ((totalBatches - globalBatch) * batchMs) /
            1000
          ).toFixed(0);

          console.log(
            `[TRAIN] ${globalBatch}/${totalBatches} | Epoch ${e + 1}/${epochs} | ` +
              `Loss: ${loss.toFixed(4)} | Batch: ${batchMs}ms | Elapsed: ${elapsedSec}s | ETA: ${etaSec}s`,
          );
        }

        console.log(
          `\n[TRAIN] Epoch ${e + 1}/${epochs} complete | Avg loss: ${(epochLoss / sequences.length).toFixed(4)}\n`,
        );
      }

      baseModel.saveWeights(BASE_WEIGHTS_PATH);
      const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(
        `[TRAIN] Done in ${totalTime}s. Saved base weights to ${BASE_WEIGHTS_PATH}`,
      );

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

      const epochs = data.epochs || 5;
      const batchSize = data.batchSize || 16;
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
          'POST /train-ghost': 'Train personality weights for a specific ghost',
          'POST /speak': 'Talk to a ghost (uses personal model + history)',
          'GET /ghosts': 'List ghosts',
          'POST /ghosts': 'Create a ghost',
          'GET /history?ghost=NAME': 'View short-term text history',
        },
        trainingTips: {
          stride: '64 = max speed, 32 = medium, 8 = slow but thorough',
          batchSize: '16 recommended for base, 8 for ghosts',
          epochs: '3-5 for base, 5-10 for ghost personality',
          warning:
            'Each batch may take 10-60s in pure JS. Use smaller batchSize for faster feedback.',
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
  console.log('\nSaving state...');
  baseTrainer.terminate();
  saveGhosts();
  process.exit(0);
});
