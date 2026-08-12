// ghost.js
const fs = require('fs');

class Ghost {
  constructor(name, backstory, model, tokenizer) {
    this.name = name;
    this.backstory = backstory;
    this.model = model;
    this.tokenizer = tokenizer;
    this.history = [];
    this.historyLimit = 10;
  }

  train(text, epochs = 5, batchSize = 16) {
    const tokens = this.tokenizer.encode(text);
    if (tokens.length < 2) return;

    const seqLen = this.model.maxSeqLen;
    const sequences = [];
    for (let i = 0; i < tokens.length - seqLen; i += seqLen) {
      sequences.push({
        inp: new Int32Array(tokens.slice(i, i + seqLen)),
        tgt: new Int32Array(tokens.slice(i + 1, i + seqLen + 1)),
      });
    }

    if (sequences.length === 0) return;

    const totalBatches = Math.ceil(sequences.length / batchSize) * epochs;
    const logEvery = Math.max(1, Math.floor(totalBatches / 15));
    let globalBatch = 0;
    let totalLoss = 0;
    let totalSteps = 0;
    const startTime = Date.now();

    console.log(
      `[TRAIN-GHOST] ${this.name}: ${epochs} epochs, ${sequences.length} sequences, batch=${batchSize}, ~${totalBatches} batches`,
    );

    for (let e = 0; e < epochs; e++) {
      let epochLoss = 0;
      let epochBatches = 0;

      for (let i = 0; i < sequences.length; i += batchSize) {
        const batch = sequences.slice(i, i + batchSize);
        const loss = this.model.trainBatch(
          batch.map((s) => s.inp),
          batch.map((s) => s.tgt),
        );

        epochLoss += loss * batch.length;
        totalLoss += loss * batch.length;
        totalSteps += batch.length;
        epochBatches++;
        globalBatch++;

        if (globalBatch % logEvery === 0) {
          const pct = ((globalBatch / totalBatches) * 100).toFixed(1);
          const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
          console.log(
            `[TRAIN-GHOST] ${this.name}: ${pct}% | Batch ${globalBatch}/${totalBatches} | Loss: ${loss.toFixed(4)} | ${elapsed}s`,
          );
        }
      }

      console.log(
        `[TRAIN-GHOST] ${this.name}: Epoch ${e + 1}/${epochs} complete | Avg loss: ${(epochLoss / (epochBatches * batchSize)).toFixed(4)}`,
      );
    }

    const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(
      `[TRAIN-GHOST] ${this.name}: Done in ${totalTime}s | Final avg loss: ${(totalLoss / totalSteps).toFixed(4)}`,
    );
  }

  learnExchange(userInput, reply) {
    const text = `User: ${userInput}\n${this.name}: ${reply}\n`;
    this.train(text, 1, 1);
  }

  speak(userInput, temperature = 0.7, learn = true) {
    let ctx = `You are ${this.name}. ${this.backstory}\n`;
    if (this.history.length) {
      ctx += this.history.join('\n') + '\n';
    }
    ctx += `User: ${userInput}\n${this.name}:`;

    let tokens = this.tokenizer.encode(ctx);
    if (tokens.length > this.model.maxSeqLen) {
      tokens = tokens.slice(-this.model.maxSeqLen);
    }

    const outTokens = this.model.generate(tokens, 120, temperature);
    let reply = this.tokenizer.decode(outTokens);

    // Hard stop at newline so the ghost doesn't hallucinate the next turn
    const newlineIdx = reply.indexOf('\n');
    if (newlineIdx !== -1) {
      reply = reply.slice(0, newlineIdx);
    }
    reply = reply.trim();

    // Fallback for untrained / empty generation
    if (!reply) {
      reply = '...';
    }

    this.history.push(`User: ${userInput}`);
    this.history.push(`${this.name}: ${reply}`);
    if (this.history.length > this.historyLimit * 2) {
      this.history = this.history.slice(-this.historyLimit * 2);
    }

    if (learn) {
      this.learnExchange(userInput, reply);
    }

    return reply;
  }

  saveWeights(path) {
    this.model.saveWeights(path);
  }

  loadWeights(path) {
    if (!fs.existsSync(path)) return;
    this.model.loadWeights(path);
  }

  toJSON() {
    return {
      name: this.name,
      backstory: this.backstory,
      history: this.history,
    };
  }

  static fromJSON(data, model, tokenizer) {
    const g = new Ghost(data.name, data.backstory, model, tokenizer);
    g.history = data.history || [];
    return g;
  }
}

module.exports = { Ghost };
