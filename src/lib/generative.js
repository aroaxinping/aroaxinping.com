/**
 * Modo generativo (opcional) de aroa-1.
 *
 * El modo por defecto de este sitio es retrieval puro: nunca inventa texto,
 * sólo devuelve el documento del corpus más parecido. Este módulo añade un
 * modo aparte, que el visitante activa a mano, donde un LLM pequeño
 * (Llama-3.2-1B-Instruct, vía WebLLM/WebGPU) redacta la respuesta usando como
 * contexto los documentos que ya ha encontrado el retrieval — RAG, no un chat
 * libre. Sigue sin haber servidor: el modelo se descarga y corre en el propio
 * navegador, así que sigue sin tener coste ni backend.
 *
 * No se carga nada de esto hasta que alguien pulsa el botón: son ~700 MB de
 * descarga y necesita WebGPU, así que no puede ser el camino por defecto.
 */

const MODELO_ID = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
const CDN_WEBLLM = 'https://esm.run/@mlc-ai/web-llm';

let motor = null;

/**
 * Si este navegador puede mover el modelo.
 *
 * No basta con mirar si `navigator.gpu` existe: hay equipos que exponen la API
 * y luego no dan ningún adaptador utilizable (GPU incompatible, WebGPU
 * desactivado por bandera). Pedir el adaptador es la única respuesta fiable, y
 * es barato comparado con enterarse a mitad de una descarga de 700 MB.
 */
export async function soportado() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false;
  try {
    return Boolean(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

/**
 * Descarga y arranca el motor. `onProgreso(informe)` recibe los eventos de
 * carga de WebLLM (útil para una barra de progreso: son varios cientos de MB).
 */
export async function cargar(onProgreso) {
  if (motor) return motor;
  if (!(await soportado())) throw new Error('este navegador no tiene WebGPU utilizable');

  const webllm = await import(/* @vite-ignore */ CDN_WEBLLM);
  motor = await webllm.CreateMLCEngine(MODELO_ID, {
    initProgressCallback: onProgreso,
  });
  return motor;
}

const SYSTEM_PROMPT = `Eres aroa-1, un modelo entrenado con el corpus personal de Aroa
Xinping (analista de datos y creadora de contenido tech). Responde SOLO con la
información de los documentos de contexto que te doy, en su mismo tono: directo,
honesto, sin relleno, con humor seco y sin presumir. Si el contexto no cubre la
pregunta, dilo abiertamente en vez de inventar. Respuestas cortas, en español.`;

/**
 * Redacta una respuesta a partir de la pregunta y de los documentos que ya
 * encontró el retrieval (RAG): el LLM no improvisa datos, sólo los reformula.
 */
export async function generar(pregunta, documentosContexto) {
  const engine = await cargar();
  const contexto = documentosContexto
    .map((d, i) => `[doc ${i + 1} — ${d.tema}] ${d.respuesta}`)
    .join('\n');

  const respuesta = await engine.chat.completions.create({
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: `Contexto:\n${contexto}\n\nPregunta: ${pregunta}` },
    ],
    temperature: 0.4,
  });

  return respuesta.choices[0]?.message?.content?.trim() ?? '';
}
