//tests/gradient_checker.js
const Transformer = require('../transformer');
const model = new Transformer(0.001, 512, 128, 4, 4, 512, 128);
const input = new Int32Array([10, 20, 30, 40, 50]);
const target = new Int32Array([20, 30, 40, 50, 60]);

// Check wte for token 10, dimension 0 (token 10 IS in the input)
const idx = 10 * 128 + 0;

model.zeroGrad();
const { loss, cache } = model.forward(input, target);
model.backward(cache);
const analytical = model.dwte[idx];

const eps = 1e-4;
model.wte[idx] += eps;
const l1 = model.forward(input, target).loss;
model.wte[idx] -= 2 * eps;
const l2 = model.forward(input, target).loss;
model.wte[idx] += eps;
const numerical = (l1 - l2) / (2 * eps);

console.log('Analytical:', analytical.toFixed(6));
console.log('Numerical: ', numerical.toFixed(6));
console.log(
  'Rel error: ',
  (
    Math.abs(analytical - numerical) /
    (Math.abs(analytical) + Math.abs(numerical) + 1e-8)
  ).toFixed(6),
);
