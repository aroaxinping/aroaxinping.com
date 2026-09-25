/**
 * Inferencia de aroa-1 en el navegador: descarga el artefacto entrenado
 * (`/model/aroa-1.json`) y responde preguntas por similitud del coseno.
 * No hay servidor: todo el cálculo ocurre aquí.
 */

import { analizar, PREFIJO_CHAR } from './analyzer.js';

const RUTA_MODELO = '/model/aroa-1.json';

let cache = null;

/**
 * Prepara un artefacto ya descargado para inferencia. Se usa desde
 * `cargarModelo()` y desde el test de paridad, que lee el JSON del disco en
 * vez de por red.
 */
export function usarModelo(datos, cargaMs = 0) {
  const indice = new Map();
  datos.terminos.forEach((t, i) => indice.set(t, i));
  cache = { ...datos, indice, cargaMs };
  return cache;
}

export async function cargarModelo() {
  if (cache) return cache;
  const t0 = performance.now();
  const res = await fetch(RUTA_MODELO);
  if (!res.ok) throw new Error(`no se pudo cargar el modelo (${res.status})`);
  const datos = await res.json();
  return usarModelo(datos, Math.round(performance.now() - t0));
}

/** tf sublineal, igual que `sublinear_tf=True` en sklearn: 1 + ln(tf). */
function vectorConsulta(modelo, texto) {
  const features = analizar(texto);
  const conteos = new Map();
  for (const f of features) conteos.set(f, (conteos.get(f) ?? 0) + 1);

  const vec = new Map(); // índice -> peso
  for (const [feature, tf] of conteos) {
    const i = modelo.indice.get(feature);
    if (i === undefined) continue; // término fuera de vocabulario: se ignora
    const tfSub = 1 + Math.log(tf);
    const peso = tfSub * modelo.idf[i] * (feature.startsWith(PREFIJO_CHAR) ? modelo.peso_char : 1);
    vec.set(i, peso);
  }

  // Normalización L2, igual que en el entrenamiento.
  let norma = 0;
  for (const v of vec.values()) norma += v * v;
  norma = Math.sqrt(norma);
  if (norma > 0) for (const [i, v] of vec) vec.set(i, v / norma);
  return vec;
}

function coseno(consulta, docSparse) {
  // Ambos vectores están L2-normalizados: el coseno es el producto escalar.
  let sim = 0;
  for (const [i, v] of consulta) {
    // docSparse es [[indice, valor], ...] ordenado por índice
    const hit = buscar(docSparse, i);
    if (hit !== undefined) sim += v * hit;
  }
  return sim;
}

function buscar(sparse, indice) {
  let lo = 0;
  let hi = sparse.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const i = sparse[mid][0];
    if (i === indice) return sparse[mid][1];
    if (i < indice) lo = mid + 1;
    else hi = mid - 1;
  }
  return undefined;
}

/**
 * Términos que más empujaron el score del documento ganador.
 *
 * El coseno es una suma de productos, así que se puede abrir término a
 * término: cada uno aporta `pesoConsulta × pesoDocumento`. Eso convierte el
 * score de número opaco en una lista de razones.
 */
function porQue(modelo, consulta, docSparse, cuantos = 8) {
  const aportes = [];
  for (const [i, peso] of consulta) {
    const pesoDoc = buscar(docSparse, i);
    if (pesoDoc === undefined) continue;
    const termino = modelo.terminos[i];
    aportes.push({
      termino: termino.startsWith(PREFIJO_CHAR) ? termino.slice(1) : termino,
      esChar: termino.startsWith(PREFIJO_CHAR),
      aporte: peso * pesoDoc,
    });
  }
  aportes.sort((a, b) => b.aporte - a.aporte);
  return aportes.slice(0, cuantos);
}

/**
 * Responde una pregunta: devuelve el documento más parecido, su score y
 * cuánto ha tardado. Si el score no llega al umbral calibrado en el
 * entrenamiento, `bajoUmbral` es true y quien llama debe usar el fallback.
 *
 * Devuelve además los `candidatos` que quedaron detrás y los `terminos` que
 * decidieron el resultado: es lo que alimenta el modo explicación.
 */
export async function preguntar(texto, { topK = 5 } = {}) {
  const modelo = await cargarModelo();
  const t0 = performance.now();

  const consulta = vectorConsulta(modelo, texto);
  const scores = modelo.vectores.map((v, i) => ({ i, score: coseno(consulta, v) }));
  scores.sort((a, b) => b.score - a.score);

  const ms = Math.round(performance.now() - t0);
  const mejor = scores[0];
  const doc = mejor ? modelo.documentos[mejor.i] : null;

  return {
    doc,
    score: Math.max(0, mejor?.score ?? 0),
    bajoUmbral: (mejor?.score ?? 0) < modelo.umbral,
    ms,
    modelo,
    candidatos: scores.slice(0, topK).map(({ i, score }) => ({
      ...modelo.documentos[i],
      score: Math.max(0, score),
    })),
    terminos: mejor ? porQue(modelo, consulta, modelo.vectores[mejor.i]) : [],
  };
}
