/**
 * Entrada por voz: Web Speech API, nativa del navegador. Gratis, sin
 * servidor, sin librería — pero de soporte desigual (bien en Chrome/Edge,
 * ausente en Firefox de escritorio), así que todo lo que la usa comprueba
 * `soportado()` antes y se calla si no está.
 */

const Reconocedor =
  typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition);

export function soportado() {
  return Boolean(Reconocedor);
}

/**
 * Escucha una sola pregunta y la devuelve como texto.
 *
 * `onEscuchando()` se llama al empezar a captar audio, útil para animar el
 * botón mientras se habla — el reconocimiento en sí no da resultados
 * parciales fiables en español, así que no se intenta transcribir en vivo.
 */
export function escuchar({ onEscuchando } = {}) {
  return new Promise((resolve, reject) => {
    if (!Reconocedor) {
      reject(new Error('este navegador no soporta reconocimiento de voz'));
      return;
    }

    const r = new Reconocedor();
    r.lang = 'es-ES';
    r.interimResults = false;
    r.maxAlternatives = 1;

    r.onspeechstart = () => onEscuchando?.();
    r.onresult = (e) => resolve(e.results[0][0].transcript);
    r.onerror = (e) => reject(new Error(e.error));
    r.onend = () => reject(new Error('sin resultado'));
    // Si sí hay resultado, onresult ya resolvió antes de que onend rechace:
    // una promesa sólo se resuelve una vez, así que esto es inofensivo.

    r.start();
  });
}
