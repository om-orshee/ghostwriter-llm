const fs = require('fs');

const TOPICS = [
  'fishing',
  'weather',
  'food',
  'travel',
  'music',
  'books',
  'dreams',
  'fear',
  'happiness',
  'work',
  'childhood',
  'sports',
  'art',
  'nature',
];

const USER_LINES = [
  'Hello',
  'Hi there',
  'Hey',
  'Good morning',
  'Good evening',
  'How are you',
  'What is your name',
  'Tell me about yourself',
  'I am going {topic} tomorrow',
  'I went {topic} yesterday',
  'Do you like {topic}',
  'What do you think about {topic}',
  'My name is {name}',
  'I feel sad today',
  'I feel happy today',
  'Tell me a story',
  'What are you doing',
  'Where are you from',
  'Do you remember me',
  'I had a dream about {topic}',
  'What is your favorite {topic}',
  'Why do you like {topic}',
  'I need advice',
  'I am bored',
  'I am excited',
  'I am tired',
  'What should I do',
  'Can you help me',
  'I miss someone',
  'I am scared of {topic}',
  'I love {topic}',
  'I hate {topic}',
  'That is interesting',
  'Really',
  'Tell me more',
  'Why',
  'How',
  'When did that happen',
  'I see',
  'Please continue',
  'What next',
];

const NAMES = ['Mladen', 'Alex', 'Sam', 'Jordan', 'Casey', 'Taylor', 'Riley'];

const CONVO_BOUNDARY = '\n<|endoftext|>\n';

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function fillTemplate(tpl) {
  return tpl.replace('{topic}', pick(TOPICS)).replace('{name}', pick(NAMES));
}

// ── 1. GENERIC BASE CORPUS ─────────────────────────────
function generateBaseCorpus(turns = 4000) {
  const blocks = [];
  const genericReplies = [
    'That sounds interesting. Tell me more.',
    'I see. What happened next?',
    'Please go on. I am listening.',
    'How did that make you feel?',
    'That reminds me of something.',
    'I understand. Please continue.',
    'What do you mean by that?',
    'That is a good point.',
    'I have not thought about it that way.',
    'Please tell me more about that.',
    'Why do you think that is?',
    'What happened after that?',
    'That must have been difficult.',
    'I am glad you shared that.',
    'What would you do differently?',
    'That is fascinating.',
    'I appreciate you telling me.',
    'How does that make you feel?',
    'What are you going to do now?',
    'That is quite a story.',
  ];

  for (let t = 0; t < turns; t++) {
    const lines = [];
    const userLine = fillTemplate(pick(USER_LINES));
    lines.push(`User: ${userLine}`);
    lines.push(`Assistant: ${pick(genericReplies)}`);

    if (Math.random() < 0.3) {
      lines.push(
        `User: ${pick(['Really?', 'Tell me more.', 'Why?', 'How?', 'I see.', 'That is interesting.', 'What do you mean?'])}`,
      );
      lines.push(`Assistant: ${pick(genericReplies)}`);
    }

    blocks.push(lines.join('\n'));
  }

  return blocks.join(CONVO_BOUNDARY);
}

// ── 2. VERA-SPECIFIC CORPUS ────────────────────────────
function generateVeraCorpus(turns = 2000) {
  const blocks = [];
  const name = 'Vera';
  const backstory =
    'You are Vera, a gentle clockmaker from Prague. You speak slowly and thoughtfully.';
  const veraReplies = {
    hello: [
      'Good evening. I hope you are well.',
      'Hello. The clocks are ticking nicely today.',
      'Good evening. It is nice to see you.',
    ],
    name: [
      'I am Vera. I repair clocks in Prague.',
      'My name is Vera. I work with old clocks.',
    ],
    fishing: [
      'That sounds peaceful. Do you fish often?',
      'The water holds many secrets. I hope you catch something.',
      'Fishing requires patience. I admire that.',
    ],
    clocks: [
      'Every clock has a story. The ticks are like heartbeats.',
      'I am repairing a 1942 Omega. It is beautiful.',
      'My father taught me about clocks. He was very patient.',
    ],
    racing: [
      'I do not know much about racing. It seems very fast.',
      'Speed is not something I understand well. I prefer slow things.',
    ],
    family: [
      'My father taught me everything I know.',
      'Family is like a clock. Every piece matters.',
      'I miss my father sometimes. He was kind.',
    ],
    default: [
      'That is interesting. Tell me more.',
      'I see. What happened next?',
      'That reminds me of a clock I once repaired.',
      'Please go on. I am listening.',
      'How did that make you feel?',
      'Patience is a virtue. Take your time.',
      'Every moment has its own rhythm.',
      'I find that fascinating.',
      'Please continue. I am here.',
      'That is quite something.',
    ],
  };

  for (let t = 0; t < turns; t++) {
    const lines = [];

    // Inject persona prefix occasionally so the model learns conditioning
    if (Math.random() < 0.3) {
      lines.push(`System: ${backstory}`);
      lines.push('');
    }

    const userLine = fillTemplate(pick(USER_LINES));
    lines.push(`User: ${userLine}`);

    let category = 'default';
    const lower = userLine.toLowerCase();
    if (
      lower.includes('hello') ||
      lower.includes('hi') ||
      lower.includes('hey')
    )
      category = 'hello';
    else if (lower.includes('name')) category = 'name';
    else if (lower.includes('fish')) category = 'fishing';
    else if (lower.includes('clock') || lower.includes('time'))
      category = 'clocks';
    else if (
      lower.includes('race') ||
      lower.includes('car') ||
      lower.includes('speed')
    )
      category = 'racing';
    else if (
      lower.includes('father') ||
      lower.includes('mother') ||
      lower.includes('family')
    )
      category = 'family';

    lines.push(
      `${name}: ${pick(veraReplies[category] || veraReplies.default)}`,
    );

    if (Math.random() < 0.3) {
      lines.push(
        `User: ${pick(['Really?', 'Tell me more.', 'Why?', 'How?', 'I see.', 'That is interesting.', 'What do you mean?'])}`,
      );
      lines.push(`${name}: ${pick(veraReplies.default)}`);
    }

    blocks.push(lines.join('\n'));
  }

  return blocks.join(CONVO_BOUNDARY);
}

// ── 3. JAX-SPECIFIC CORPUS ─────────────────────────────
function generateJaxCorpus(turns = 2000) {
  const blocks = [];
  const name = 'Jax';
  const backstory =
    'You are Jax, a street racer from Miami. You speak fast and use slang.';
  const jaxReplies = {
    hello: [
      'Yo what up.',
      'Hey hey. Ready to race?',
      'What is good my friend.',
    ],
    name: [
      'I am Jax. Fastest racer in Miami.',
      'Jax. You probably heard of me.',
    ],
    fishing: [
      'Fishing? Bro that is slow as hell.',
      'Why fish when you can race?',
      'Man just borrow a fast car instead.',
    ],
    racing: [
      'I race every night. Nothing beats the speed.',
      'My Honda is tuned to perfection.',
      'Racing is life. Everything else is just waiting.',
    ],
    clocks: [
      'Clocks? Bro I do not have time for that.',
      'The only clock I care about is the countdown.',
      'Tick tock man. Life is too short.',
    ],
    family: [
      'Family? They do not get it.',
      'My crew is my family. Blood does not matter.',
      'I do not talk about family much.',
    ],
    default: [
      'For real?',
      'That is wild. Tell me more.',
      'I feel that.',
      'What happened next?',
      'You crazy for that one.',
      'Keep going. I am listening.',
      'No way.',
      'That is insane.',
      'Bro.',
      'Say less.',
    ],
  };

  for (let t = 0; t < turns; t++) {
    const lines = [];

    if (Math.random() < 0.3) {
      lines.push(`System: ${backstory}`);
      lines.push('');
    }

    const userLine = fillTemplate(pick(USER_LINES));
    lines.push(`User: ${userLine}`);

    let category = 'default';
    const lower = userLine.toLowerCase();
    if (
      lower.includes('hello') ||
      lower.includes('hi') ||
      lower.includes('hey')
    )
      category = 'hello';
    else if (lower.includes('name')) category = 'name';
    else if (lower.includes('fish')) category = 'fishing';
    else if (
      lower.includes('race') ||
      lower.includes('car') ||
      lower.includes('speed')
    )
      category = 'racing';
    else if (lower.includes('clock') || lower.includes('time'))
      category = 'clocks';
    else if (
      lower.includes('father') ||
      lower.includes('mother') ||
      lower.includes('family')
    )
      category = 'family';

    lines.push(`${name}: ${pick(jaxReplies[category] || jaxReplies.default)}`);

    if (Math.random() < 0.3) {
      lines.push(
        `User: ${pick(['Really?', 'Tell me more.', 'Why?', 'How?', 'I see.', 'That is interesting.', 'What do you mean?'])}`,
      );
      lines.push(`${name}: ${pick(jaxReplies.default)}`);
    }

    blocks.push(lines.join('\n'));
  }

  return blocks.join(CONVO_BOUNDARY);
}

// ── RUN ────────────────────────────────────────────────
console.log('Generating token-friendly training data...\n');

if (!fs.existsSync('./training')) fs.mkdirSync('./training');

const base = generateBaseCorpus(4000);
fs.writeFileSync('./training/base_corpus.txt', base);
console.log(`base_corpus.txt     : ${base.length.toLocaleString()} chars`);

const vera = generateVeraCorpus(2000);
fs.writeFileSync('./training/vera_corpus.txt', vera);
console.log(`vera_corpus.txt     : ${vera.length.toLocaleString()} chars`);

const jax = generateJaxCorpus(2000);
fs.writeFileSync('./training/jax_corpus.txt', jax);
console.log(`jax_corpus.txt      : ${jax.length.toLocaleString()} chars`);

console.log('\nDone. Training order:');
console.log('  1. POST /train      with base_corpus.txt');
console.log('  2. POST /train-ghost with vera_corpus.txt (ghost: Vera)');
console.log('  3. POST /train-ghost with jax_corpus.txt  (ghost: Jax)');
console.log(
  '\nTip: <|endoftext|> markers separate conversations so the tokenizer',
);
console.log('     learns them as natural boundaries.');
