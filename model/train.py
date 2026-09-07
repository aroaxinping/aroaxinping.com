"""Entrena aroa-1 y lo exporta a un JSON que corre en el navegador.

    python model/train.py

Lee `corpus.jsonl`, vectoriza con TF-IDF (scikit-learn), calibra el umbral de
confianza con `eval.jsonl` y escribe `public/model/aroa-1.json`.

El modelo es de recuperación, no generativo: no inventa texto, devuelve el
documento del corpus más parecido a la pregunta. La gracia de que quepa en un
JSON es que no hace falta servidor ni API — la inferencia ocurre entera en el
navegador de quien visita la web.
"""

import json
import statistics
from datetime import date
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from analyzer import PESO_CHAR, PREFIJO_CHAR, analizar

AQUI = Path(__file__).parent
CORPUS = AQUI / "corpus.jsonl"
EVAL = AQUI / "eval.jsonl"
SALIDA = AQUI.parent / "public" / "model" / "aroa-1.json"
SALIDA_EVAL = AQUI.parent / "public" / "model" / "evaluacion.json"

# Las preguntas de ejemplo pesan más que la respuesta: quien escribe en el chat
# escribe preguntas, no párrafos, y el vector del documento debe parecerse a eso.
# Repetirlas más de dos veces empieza a tapar el contenido de la respuesta.
PESO_PREGUNTAS = 2

VERSION = "1.0.0"

# Peso por debajo del cual un término se descarta al exportar.
#
# Barrido sobre el set de evaluación: 0.0 / 0.005 / 0.02 dan 97.6% y 0.01 da
# 95.2%. Que no sea monótono ya dice lo que hay — a estas alturas la diferencia
# es de una pregunta en 42, o sea ruido, no señal. Así que se elige por el
# criterio que sí es medible: 0.02 recorta el artefacto de 578 a 414 KB.
PODA = 0.02


def leer_jsonl(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def texto_documento(doc: dict) -> str:
    return " ".join([" ".join(doc["preguntas"])] * PESO_PREGUNTAS + [doc["respuesta"]])


def calibrar_umbral(pos: list[float], neg: list[float]) -> tuple[float, float, float]:
    """Punto de corte entre lo que sé contestar y lo que no.

    Los dos grupos se solapan, así que no hay un corte perfecto: o rechazo
    preguntas que sí sabía, o contesto preguntas que no. Se barren todos los
    cortes posibles y se elige el que mejor equilibra ambos errores (media de
    aciertos en dominio y fuera de dominio), con un pequeño sesgo hacia callarse:
    equivocarse en silencio es más barato que equivocarse en voz alta.

    Devuelve (umbral, % de preguntas en dominio aceptadas, % de fuera rechazadas).
    """
    if not pos or not neg:
        return 0.15, 1.0, 1.0

    candidatos = sorted({round(s, 4) for s in pos + neg})
    mejor = (0.0, 0.15, 0.0, 0.0)
    for u in candidatos:
        aceptadas = sum(1 for s in pos if s >= u) / len(pos)
        rechazadas = sum(1 for s in neg if s < u) / len(neg)
        marca = (aceptadas + rechazadas) / 2 + rechazadas * 0.01
        if marca > mejor[0]:
            mejor = (marca, u, aceptadas, rechazadas)
    return mejor[1], mejor[2], mejor[3]


def recortar(X, minimo: float):
    """Tira los pesos que no mueven la aguja, para adelgazar el artefacto."""
    X = X.copy()
    X.data[X.data <= minimo] = 0
    X.eliminate_zeros()
    return X


def matriz_por_tema(resultados: list[dict], temas: list[str]) -> list[list[int]]:
    """Matriz de confusión agregada por tema.

    Por documento serían 50×50 y no habría quien la leyera; por tema son nueve
    filas y se ve de un vistazo qué áreas se confunden entre sí.
    """
    pos = {t: i for i, t in enumerate(temas)}
    matriz = [[0] * len(temas) for _ in temas]
    for r in resultados:
        if not r["en_dominio"]:
            continue
        matriz[pos[r["tema_esperado"]]][pos[r["tema_obtenido"]]] += 1
    return matriz


def histograma(resultados: list[dict], paso: float = 0.05) -> dict:
    """Reparto de scores dentro y fuera de dominio, para dibujar el solape.

    Es la imagen que justifica el umbral: se ve dónde se pisan las dos
    distribuciones y qué se pierde al cortar por ahí.
    """
    tope = max((r["score"] for r in resultados), default=0.0)
    n = max(1, int(tope / paso) + 1)
    bins = [round(i * paso, 2) for i in range(n)]
    dentro = [0] * n
    fuera = [0] * n
    for r in resultados:
        i = min(int(r["score"] / paso), n - 1)
        (dentro if r["en_dominio"] else fuera)[i] += 1
    return {"paso": paso, "bins": bins, "en_dominio": dentro, "fuera": fuera}


def main() -> None:
    docs = leer_jsonl(CORPUS)
    pruebas = leer_jsonl(EVAL)

    # norm=None porque la normalización se hace después de pesar el bloque de
    # caracteres; si normalizara antes, el peso se perdería.
    vectorizador = TfidfVectorizer(
        analyzer=analizar,
        sublinear_tf=True,
        smooth_idf=True,
        norm=None,
        min_df=1,
    )
    X = vectorizador.fit_transform(texto_documento(d) for d in docs)

    terminos = vectorizador.get_feature_names_out()
    pesos = sp.diags(
        [PESO_CHAR if t.startswith(PREFIJO_CHAR) else 1.0 for t in terminos]
    )
    X = normalize(X @ pesos)

    # Los vectores se podan y se redondean para que el JSON no pese de más, y
    # es esa versión —no la exacta— la que corre en el navegador. Así que la
    # evaluación se hace sobre ella: las métricas tienen que describir el
    # modelo que se sirve, no un ideal que nadie llega a usar.
    X = np.round(normalize(recortar(X, PODA)).toarray(), 4)

    # ── Evaluación ────────────────────────────────────────────────────────────
    Q = normalize(vectorizador.transform(p["pregunta"] for p in pruebas) @ pesos)
    sims = cosine_similarity(Q, X)
    mejores = sims.argmax(axis=1)
    scores = sims.max(axis=1)

    tema_de = {d["id"]: d["tema"] for d in docs}

    resultados = []
    for p, idx, score in zip(pruebas, mejores, scores):
        obtenido = docs[idx]["id"]
        en_dominio = p["esperado"] is not None
        resultados.append(
            {
                "pregunta": p["pregunta"],
                "esperado": p["esperado"],
                "obtenido": obtenido,
                "tema_esperado": tema_de.get(p["esperado"]),
                "tema_obtenido": tema_de[obtenido],
                "score": round(float(score), 4),
                "en_dominio": en_dominio,
                "acierto": (obtenido == p["esperado"]) if en_dominio else None,
            }
        )

    pos_scores = [r["score"] for r in resultados if r["en_dominio"]]
    neg_scores = [r["score"] for r in resultados if not r["en_dominio"]]
    aciertos = sum(1 for r in resultados if r["acierto"])
    fallos = [
        (r["pregunta"], r["esperado"], r["obtenido"], r["score"])
        for r in resultados
        if r["acierto"] is False
    ]

    total_pos = len(pos_scores)
    accuracy = aciertos / total_pos if total_pos else 0.0
    umbral, cobertura, rechazo = calibrar_umbral(pos_scores, neg_scores)

    # ── Exportación ───────────────────────────────────────────────────────────
    idf = vectorizador.idf_
    # X ya viene podada y redondeada de arriba: se exporta tal cual, para que
    # el navegador calcule exactamente sobre los mismos números con los que se
    # ha evaluado. `scripts/paridad.mjs` comprueba que así sea.
    vectores = [
        [[int(i), float(fila[i])] for i in np.nonzero(fila)[0]] for fila in X
    ]

    artefacto = {
        "nombre": "aroa-1",
        "version": VERSION,
        "entrenado": date.today().isoformat(),
        "arquitectura": "TF-IDF (palabras 1-2 + caracteres 3-5) + similitud del coseno",
        "umbral": umbral,
        "peso_char": PESO_CHAR,
        "metricas": {
            "documentos": len(docs),
            "vocabulario": len(terminos),
            "preguntas_evaluadas": total_pos,
            "accuracy_top1": round(accuracy, 4),
            "fuera_de_dominio": len(neg_scores),
            "cobertura": round(cobertura, 4),
            "rechazo_fuera_dominio": round(rechazo, 4),
        },
        "terminos": list(terminos),
        "idf": [round(float(v), 4) for v in idf],
        "vectores": vectores,
        "documentos": [
            {
                "id": d["id"],
                "tema": d["tema"],
                "respuesta": d["respuesta"],
                "preguntas": d["preguntas"],
            }
            for d in docs
        ],
    }

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(
        json.dumps(artefacto, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # Informe de evaluación aparte: la web lo publica entero en /eval. Enseñar
    # dónde falla el modelo vale más que decir que funciona.
    temas = list(dict.fromkeys(d["tema"] for d in docs))
    informe = {
        "modelo": "aroa-1",
        "version": VERSION,
        "entrenado": artefacto["entrenado"],
        "umbral": umbral,
        "metricas": artefacto["metricas"],
        "temas": temas,
        "matriz": matriz_por_tema(resultados, temas),
        "histograma": histograma(resultados),
        "resultados": resultados,
    }
    SALIDA_EVAL.write_text(
        json.dumps(informe, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # ── Informe ───────────────────────────────────────────────────────────────
    peso = SALIDA.stat().st_size / 1024
    print(f"documentos      {len(docs)}")
    print(f"vocabulario     {len(terminos)} features")
    print(f"accuracy@1      {accuracy:.1%}  ({aciertos}/{total_pos})")
    if pos_scores:
        print(f"score en dominio   min {min(pos_scores):.3f}  mediana {statistics.median(pos_scores):.3f}")
    if neg_scores:
        print(f"score fuera        max {max(neg_scores):.3f}  mediana {statistics.median(neg_scores):.3f}")
    print(f"umbral          {umbral}  (acepta {cobertura:.0%} en dominio, rechaza {rechazo:.0%} de fuera)")
    print(f"artefacto       {SALIDA.relative_to(AQUI.parent)}  ({peso:.0f} KB)")
    print(f"informe         {SALIDA_EVAL.relative_to(AQUI.parent)}")
    if fallos:
        print("\nfallos:")
        for pregunta, esperado, obtenido, score in fallos:
            print(f"  «{pregunta}»\n    esperaba {esperado}, devolvió {obtenido} ({score:.3f})")


if __name__ == "__main__":
    main()
