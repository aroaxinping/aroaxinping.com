"""Analizador de texto de aroa-1.

Este fichero es la mitad de un contrato: `src/lib/infer.js` implementa
exactamente lo mismo en JavaScript. Si tocas uno, toca el otro — si dejan de
coincidir, el vector de la consulta cae en un espacio distinto al de los
documentos y el modelo devuelve basura con toda la confianza del mundo.

Se usa un analizador propio en vez de los de sklearn precisamente por eso:
replicar `char_wb` de sklearn a ciegas es frágil, y aquí las reglas están
escritas donde se pueden leer.
"""

import re
import unicodedata

TOKEN = re.compile(r"\w\w+", re.UNICODE)
ESPACIOS = re.compile(r"\s+", re.UNICODE)

WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (3, 5)

# Los n-gramas de caracteres son muchísimos más que las palabras, así que sin
# frenarlos se comen el vector y todo empieza a parecerse a todo. A 0.3 aportan
# tolerancia a erratas sin decidir ellos solos la respuesta. El número no es de
# intuición: sale de probar 0.3 / 0.5 / 1.0 contra el set de evaluación.
PESO_CHAR = 0.3

# Sin lista de palabras vacías: quitarlas bajó la accuracy del 88% al 74%.
# Tiene sentido — aquí "quién eres" o "por qué" no son ruido, son justo lo que
# distingue una pregunta de otra.


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y con los espacios colapsados.

    Quitar las tildes es lo que hace que "que estudiaste" encuentre lo mismo que
    "qué estudiaste". Nadie escribe con tildes en un chat.
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = unicodedata.normalize("NFC", texto)
    return ESPACIOS.sub(" ", texto).strip()


def _tokens(texto: str) -> list[str]:
    return TOKEN.findall(texto)


def _palabras(texto: str) -> list[str]:
    tokens = _tokens(texto)
    grams = list(tokens)
    for n in range(WORD_NGRAMS[0] + 1, WORD_NGRAMS[1] + 1):
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))
    return grams


def _caracteres(texto: str) -> list[str]:
    """n-gramas de caracteres acotados a la palabra, al estilo `char_wb`.

    Son los que hacen que el modelo entienda "que estudias" y "qué estudiaste",
    o que sobreviva a una errata. En castellano valen más que cualquier stemmer.
    """
    grams = []
    for token in _tokens(texto):
        w = f" {token} "
        for n in range(CHAR_NGRAMS[0], CHAR_NGRAMS[1] + 1):
            if len(w) <= n:
                grams.append(w)
                break
            for i in range(len(w) - n + 1):
                grams.append(w[i : i + n])
    return grams


# Marca los features de caracteres para poder pesarlos aparte más tarde.
PREFIJO_CHAR = "\x00"


def analizar(texto: str) -> list[str]:
    """Texto crudo -> lista de features. Es lo que recibe el vectorizador."""
    texto = normalizar(texto)
    return _palabras(texto) + [PREFIJO_CHAR + g for g in _caracteres(texto)]
