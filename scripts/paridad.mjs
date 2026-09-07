/**
 * Test de paridad Python ↔ JavaScript.
 *
 *     node scripts/paridad.mjs
 *
 * El modelo se entrena en Python y se ejecuta en JavaScript, así que hay dos
 * implementaciones del mismo analizador que tienen que coincidir carácter a
 * carácter. Cuando dejan de hacerlo no salta ningún error: el vector de la
 * consulta cae en otro espacio y el modelo empieza a devolver documentos
 * equivocados con toda la confianza del mundo.
 *
 * Este test pasa el set de evaluación entero por la implementación JS y lo
 * compara con lo que dejó escrito `train.py`. Existe porque un `if` de más en
 * `coseno()` estuvo devolviendo score 0 para preguntas largas contra
 * documentos cortos, y nada lo delató.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { usarModelo, preguntar } from '../src/lib/infer.js';

const RAIZ = join(dirname(fileURLToPath(import.meta.url)), '..');
const TOLERANCIA = 1e-3; // los vectores se exportan redondeados a 4 decimales

const leer = (ruta) => JSON.parse(readFileSync(join(RAIZ, ruta), 'utf-8'));

const modelo = leer('public/model/aroa-1.json');
const informe = leer('public/model/evaluacion.json');

usarModelo(modelo);

let fallos = 0;
let maxDelta = 0;

for (const esperado of informe.resultados) {
  const obtenido = await preguntar(esperado.pregunta);
  const delta = Math.abs(obtenido.score - esperado.score);
  maxDelta = Math.max(maxDelta, delta);

  if (obtenido.doc.id !== esperado.obtenido) {
    fallos++;
    console.error(
      `documento distinto  «${esperado.pregunta}»\n` +
        `  python: ${esperado.obtenido} (${esperado.score})\n` +
        `  js:     ${obtenido.doc.id} (${obtenido.score.toFixed(4)})`,
    );
  } else if (delta > TOLERANCIA) {
    fallos++;
    console.error(
      `score distinto      «${esperado.pregunta}»\n` +
        `  python: ${esperado.score}  js: ${obtenido.score.toFixed(4)}  Δ ${delta.toFixed(5)}`,
    );
  }
}

const total = informe.resultados.length;
if (fallos) {
  console.error(`\n✗ ${fallos}/${total} discrepancias — analyzer.js e analyzer.py NO coinciden`);
  process.exit(1);
}

console.log(`✓ paridad python↔js  ${total}/${total} preguntas  ·  Δscore máx ${maxDelta.toFixed(6)}`);
