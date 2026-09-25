/**
 * Comandos del chat.
 *
 * El input de aroa-1 hace de terminal además de chat: lo que empieza por `/`
 * no va al modelo, se ejecuta. Sale gratis en espacio de pantalla y quien
 * quiera hurgar en el corpus puede hacerlo sin salir de la conversación.
 *
 * Cada comando recibe un contexto con lo que necesita para hacer su trabajo y
 * devuelve el texto a pintar (o nada, si ya ha actuado por su cuenta).
 */

export const COMANDOS = {
  ayuda: {
    args: '',
    desc: 'esta lista',
    run: () =>
      [
        'comandos disponibles:',
        ...Object.entries(COMANDOS).map(([nombre, c]) => {
          const etiqueta = `/${nombre}${c.args ? ` ${c.args}` : ''}`;
          return `  ${etiqueta.padEnd(18)}${c.desc}`;
        }),
        '',
        'todo lo que no empiece por / se lo come el modelo.',
      ].join('\n'),
  },

  corpus: {
    args: '[tema]',
    desc: 'lista los documentos del corpus',
    run: ({ modelo, arg }) => {
      const docs = arg
        ? modelo.documentos.filter((d) => d.tema === arg)
        : modelo.documentos;

      if (!docs.length) {
        const temas = [...new Set(modelo.documentos.map((d) => d.tema))];
        return `no hay ningún documento con el tema «${arg}».\ntemas: ${temas.join(', ')}`;
      }

      const porTema = {};
      for (const d of docs) (porTema[d.tema] ??= []).push(d.id);

      return [
        `${docs.length} documento${docs.length === 1 ? '' : 's'}${arg ? ` en «${arg}»` : ''}:`,
        ...Object.entries(porTema).map(
          ([tema, ids]) => `\n  ${tema}\n${ids.map((i) => `    ${i}`).join('\n')}`,
        ),
        '\nusa /doc <id> para ver uno entero.',
      ].join('\n');
    },
  },

  doc: {
    args: '<id>',
    desc: 'muestra un documento tal cual',
    run: ({ modelo, arg }) => {
      if (!arg) return 'dime cuál: /doc <id>. Con /corpus los tienes todos.';
      const doc = modelo.documentos.find((d) => d.id === arg);
      if (!doc) return `no tengo ningún documento con id «${arg}».`;
      return [
        `id:        ${doc.id}`,
        `tema:      ${doc.tema}`,
        `preguntas: ${doc.preguntas.length} variantes de entrenamiento`,
        '',
        doc.respuesta,
      ].join('\n');
    },
  },

  modelo: {
    args: '',
    desc: 'ficha técnica de la variante activa',
    run: ({ modelo, variante }) => {
      const m = modelo.metricas;
      return [
        `nombre:       ${modelo.nombre} · ${variante}`,
        `versión:      ${modelo.version}`,
        `arquitectura: ${modelo.arquitectura}`,
        `entrenado:    ${modelo.entrenado}`,
        `documentos:   ${m.documentos}`,
        `vocabulario:  ${m.vocabulario} términos`,
        `accuracy@1:   ${Math.round(m.accuracy_top1 * 100)}%`,
        `umbral:       ${modelo.umbral}`,
        '',
        'el detalle completo está en /model-card y /eval.',
      ].join('\n');
    },
  },

  debug: {
    args: '',
    desc: 'explicar o no cada respuesta',
    run: ({ alternarDebug }) =>
      `modo explicación ${alternarDebug() ? 'activado' : 'desactivado'}.`,
  },

  tema: {
    args: '',
    desc: 'claro / oscuro',
    run: () => {
      document.getElementById('temaBtn')?.click();
      return `tema ${document.documentElement.dataset.theme === 'light' ? 'claro' : 'oscuro'}.`;
    },
  },

  limpiar: {
    args: '',
    desc: 'vacía la conversación',
    run: ({ limpiar }) => {
      limpiar();
      return null;
    },
  },

  eval: {
    args: '',
    desc: 'abre la evaluación completa',
    run: () => {
      location.href = '/eval';
      return null;
    },
  },
};

export const esComando = (texto) => texto.startsWith('/');

/** Nombres que encajan con lo tecleado, para el autocompletado. */
export function sugerencias(texto) {
  if (!esComando(texto)) return [];
  const escrito = texto.slice(1).split(' ')[0].toLowerCase();
  if (texto.includes(' ')) return []; // ya está escribiendo el argumento
  return Object.keys(COMANDOS).filter((n) => n.startsWith(escrito));
}

/**
 * Ejecuta lo tecleado. Devuelve el texto a pintar, o null si el comando ya se
 * ha ocupado de todo.
 */
export function ejecutar(texto, ctx) {
  const [nombre, ...resto] = texto.slice(1).trim().split(/\s+/);
  const comando = COMANDOS[nombre.toLowerCase()];
  if (!comando) {
    return `«/${nombre}» no existe. Prueba /ayuda.`;
  }
  return comando.run({ ...ctx, arg: resto.join(' ').trim() });
}
