const https = require('https');
const fs = require('fs');
const path = require('path');

const OUT_DIR = './training';
const OUT_FILE = path.join(OUT_DIR, 'corpus.txt');

const BOOKS = [
  { id: 1342, name: 'pride_and_prejudice.txt' },
  { id: 11, name: 'alice_in_wonderland.txt' },
  { id: 1661, name: 'sherlock_holmes.txt' },
  { id: 84, name: 'frankenstein.txt' },
  { id: 2701, name: 'moby_dick.txt' },
  { id: 98, name: 'tale_of_two_cities.txt' },
  { id: 174, name: 'dorian_gray.txt' },
  { id: 43, name: 'dr_jekyll.txt' },
];

const CONVERSATION_SEED = `User: Hello.
Vera: Good evening. I hope you are well.
User: What is your name?
Vera: I am Vera. I repair clocks in Prague.
User: That sounds interesting.
Vera: Every clock has a story. The ticks are like heartbeats.
User: I am going fishing tomorrow.
Vera: That sounds peaceful. Do you fish often?
User: Not really. It is my first time.
Vera: I remember my first time repairing a clock. I was nervous.
User: Hello.
Jax: Yo what up.
User: What do you do?
Jax: I race. Street racing in Miami. Fastest dude out here.
User: Is that dangerous?
Jax: Dangerous is the point, man.
User: My name is Mladen.
Jax: Mladen? Tight name. I'm Jax.
User: I am going fishing tomorrow.
Jax: Fishing? Bro that's slow as hell. You should race instead.
User: I don't have a car.
Jax: Borrow one. Live a little.`;

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} for ${url}`));
          return;
        }
        res.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      })
      .on('error', reject);
  });
}

function stripGutenbergBoilerplate(text) {
  const startMatch = text.match(
    /\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*/is,
  );
  let body = startMatch
    ? text.slice(startMatch.index + startMatch[0].length)
    : text;

  const endMatch = body.match(
    /\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*/is,
  );
  if (endMatch) body = body.slice(0, endMatch.index);

  return body
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

async function main() {
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  const parts = [];
  let totalChars = 0;

  console.log('Downloading books from Project Gutenberg...\n');

  for (const book of BOOKS) {
    const url = `https://www.gutenberg.org/files/${book.id}/${book.id}-0.txt`;
    const dest = path.join(OUT_DIR, book.name);

    try {
      if (fs.existsSync(dest)) {
        console.log(`  [cached] ${book.name}`);
      } else {
        process.stdout.write(`  [downloading] ${book.name} ... `);
        await download(url, dest);
        console.log('done');
      }

      const raw = fs.readFileSync(dest, 'utf8');
      const clean = stripGutenbergBoilerplate(raw);
      parts.push(clean);
      totalChars += clean.length;
      console.log(`  -> ${clean.length.toLocaleString()} chars cleaned`);
    } catch (err) {
      console.error(`  FAILED: ${book.name} — ${err.message}`);
    }
  }

  parts.push(CONVERSATION_SEED);
  totalChars += CONVERSATION_SEED.length;

  fs.writeFileSync(OUT_FILE, parts.join('\n\n\n'), 'utf8');

  console.log('\n-----------------------------');
  console.log(`Total clean text: ${totalChars.toLocaleString()} characters`);
  console.log(`Output file:      ${OUT_FILE}`);
  console.log(`Books included:   ${parts.length - 1}`);
  console.log('-----------------------------');
  console.log('\nNext step: run the training endpoint with corpus.txt');
}

main().catch(console.error);
