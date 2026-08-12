// tokenizer.js
const fs = require('fs');

class BPETokenizer {
  constructor(vocabSize = 512) {
    this.vocabSize = vocabSize;
    this.vocab = new Array(vocabSize); // id -> Uint8Array
    this.merges = []; // [[id1, id2], ...] in training order
    this.initBaseVocab();
  }

  initBaseVocab() {
    for (let i = 0; i < 256; i++) {
      this.vocab[i] = new Uint8Array([i]);
    }
  }

  train(text, numMerges = null) {
    if (!text || text.length === 0) {
      console.log(
        '[TOKENIZER] No text provided; using base 256-byte vocab only.',
      );
      return;
    }
    if (numMerges === null) numMerges = this.vocabSize - 256;
    numMerges = Math.min(numMerges, this.vocabSize - 256);

    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    let tokens = Array.from(bytes);

    for (let m = 0; m < numMerges; m++) {
      const counts = new Map(); // key = (id1 << 16) | id2
      for (let i = 0; i < tokens.length - 1; i++) {
        const key = (tokens[i] << 16) | tokens[i + 1];
        counts.set(key, (counts.get(key) || 0) + 1);
      }

      if (counts.size === 0) break;

      let bestKey = null;
      let bestCount = 0;
      for (const [key, count] of counts) {
        if (count > bestCount) {
          bestCount = count;
          bestKey = key;
        }
      }

      if (bestCount < 2) break;

      const id1 = (bestKey >> 16) & 0xffff;
      const id2 = bestKey & 0xffff;
      const newId = 256 + m;

      const merged = new Uint8Array(
        this.vocab[id1].length + this.vocab[id2].length,
      );
      merged.set(this.vocab[id1], 0);
      merged.set(this.vocab[id2], this.vocab[id1].length);
      this.vocab[newId] = merged;
      this.merges.push([id1, id2]);

      // Apply merge to token sequence
      const newTokens = [];
      let i = 0;
      while (i < tokens.length) {
        if (
          i < tokens.length - 1 &&
          tokens[i] === id1 &&
          tokens[i + 1] === id2
        ) {
          newTokens.push(newId);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i++;
        }
      }
      tokens = newTokens;
    }

    console.log(
      `[TOKENIZER] Trained ${this.merges.length} merges. Effective vocab: ${256 + this.merges.length}`,
    );
  }

  encode(text) {
    if (!text || text.length === 0) return [];
    const encoder = new TextEncoder();
    let tokens = Array.from(encoder.encode(text));

    for (let m = 0; m < this.merges.length; m++) {
      const [id1, id2] = this.merges[m];
      const newId = 256 + m;
      const newTokens = [];
      let i = 0;
      while (i < tokens.length) {
        if (
          i < tokens.length - 1 &&
          tokens[i] === id1 &&
          tokens[i + 1] === id2
        ) {
          newTokens.push(newId);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i++;
        }
      }
      tokens = newTokens;
    }
    return tokens;
  }

  decode(ids) {
    if (!ids || ids.length === 0) return '';
    const bytes = [];
    for (const id of ids) {
      if (this.vocab[id]) {
        for (let i = 0; i < this.vocab[id].length; i++) {
          bytes.push(this.vocab[id][i]);
        }
      }
    }
    const decoder = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true });
    return decoder.decode(new Uint8Array(bytes));
  }

  save(path) {
    const payload = {
      vocabSize: this.vocabSize,
      merges: this.merges,
      vocabBytes: this.vocab
        .slice(256, 256 + this.merges.length)
        .map((v) => Array.from(v)),
    };
    fs.writeFileSync(path, JSON.stringify(payload, null, 2));
  }

  load(path) {
    const payload = JSON.parse(fs.readFileSync(path, 'utf8'));
    this.vocabSize = payload.vocabSize;
    this.merges = payload.merges;
    this.initBaseVocab();
    for (let i = 0; i < payload.vocabBytes.length; i++) {
      this.vocab[256 + i] = new Uint8Array(payload.vocabBytes[i]);
    }
  }
}

module.exports = { BPETokenizer };
