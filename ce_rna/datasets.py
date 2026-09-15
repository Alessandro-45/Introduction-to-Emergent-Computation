"""
Conjuntos de patrones de entrenamiento usados en los experimentos.

Todos los conjuntos provienen directamente de los ejemplos de las clases:

* Compuertas logicas AND / OR (y sus parientes XOR, NAND, NOR) -> `claseRN02.md`
  y `ICE-claseRN03.md`.
* Reconocimiento de las letras X y O con un perceptron de dos neuronas de
  salida -> `ICE-claseRN03.md`, figura "Funcionamiento de un Perceptron".
* Descodificador de binario a decimal de 3 bits -> `ICE-claseRN04.md`, ejemplo
  propuesto para el ADALINE.

Convenciones
------------
`X` es siempre una matriz (N, m0) con un patron por fila y SIN la columna de
sesgo: cada modelo decide si antepone la entrada fija x0 = 1 (la "neurona de
inclinacion" de `ICE-claseRN03.md`).

`d` es un vector (N,) para problemas de una sola salida o una matriz (N, m1)
cuando hay varias neuronas de salida.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Conjunto:
    """Contenedor simple de un conjunto de entrenamiento."""

    X: np.ndarray
    d: np.ndarray
    nombre: str
    nombres_entradas: list[str] = field(default_factory=list)
    nombres_salidas: list[str] = field(default_factory=list)
    descripcion: str = ""

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.d = np.asarray(self.d, dtype=float)
        if self.X.shape[0] != self.d.shape[0]:
            raise ValueError("X y d deben tener el mismo numero de filas")
        if not self.nombres_entradas:
            self.nombres_entradas = [f"x{i + 1}" for i in range(self.X.shape[1])]
        if not self.nombres_salidas:
            n_sal = 1 if self.d.ndim == 1 else self.d.shape[1]
            self.nombres_salidas = ["y"] if n_sal == 1 else [f"y{j + 1}" for j in range(n_sal)]

    @property
    def n_patrones(self) -> int:
        return self.X.shape[0]

    @property
    def n_entradas(self) -> int:
        return self.X.shape[1]

    @property
    def n_salidas(self) -> int:
        return 1 if self.d.ndim == 1 else self.d.shape[1]

    def tabla(self):
        """Devuelve el conjunto como `pandas.DataFrame` (tabla de verdad)."""
        import pandas as pd

        datos = {nombre: self.X[:, i] for i, nombre in enumerate(self.nombres_entradas)}
        if self.d.ndim == 1:
            datos[self.nombres_salidas[0]] = self.d
        else:
            for j, nombre in enumerate(self.nombres_salidas):
                datos[nombre] = self.d[:, j]
        return pd.DataFrame(datos)


# ---------------------------------------------------------------------------
# Compuertas logicas
# ---------------------------------------------------------------------------

#: Tabla de verdad de las compuertas en codificacion binaria {0, 1}.
#: El orden de los patrones es (0,0), (0,1), (1,0), (1,1).
_ENTRADAS_BINARIAS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

_SALIDAS = {
    "AND": np.array([0, 0, 0, 1], dtype=float),
    "OR": np.array([0, 1, 1, 1], dtype=float),
    "XOR": np.array([0, 1, 1, 0], dtype=float),
    "NAND": np.array([1, 1, 1, 0], dtype=float),
    "NOR": np.array([1, 0, 0, 0], dtype=float),
}


def compuerta(nombre: str, codificacion: str = "binaria") -> Conjunto:
    """Conjunto de entrenamiento de una compuerta logica de dos entradas.

    Parameters
    ----------
    nombre:
        "AND", "OR", "XOR", "NAND" o "NOR".
    codificacion:
        ``"binaria"`` usa valores {0, 1} (la convencion de McCulloch-Pitts en
        `claseRN02.md`); ``"bipolar"`` usa {-1, +1}, que es la convencion con la
        que trabaja el perceptron bipolar de `ICE-claseRN03.md` y la regla de
        Hebb ("supongamos que xi e yi son bipolares o antisimetricas").

    Notas
    -----
    La codificacion importa mas de lo que parece.  Con entradas binarias, un
    patron (0, 0) no puede modificar ningun peso salvo el del sesgo, porque el
    incremento es proporcional a la entrada.  Con codificacion bipolar todas las
    entradas son +-1 y por tanto *todos* los pesos reciben correccion en cada
    patron: la regla de Hebb solo resuelve el AND en codificacion bipolar.
    """
    nombre = nombre.upper()
    if nombre not in _SALIDAS:
        raise KeyError(f"Compuerta desconocida '{nombre}'. Disponibles: {sorted(_SALIDAS)}")

    X = _ENTRADAS_BINARIAS.copy()
    d = _SALIDAS[nombre].copy()

    if codificacion == "bipolar":
        X = 2.0 * X - 1.0
        d = 2.0 * d - 1.0
    elif codificacion != "binaria":
        raise ValueError("codificacion debe ser 'binaria' o 'bipolar'")

    return Conjunto(
        X=X,
        d=d,
        nombre=f"{nombre} ({codificacion})",
        nombres_entradas=["x1", "x2"],
        nombres_salidas=["y"],
        descripcion=f"Tabla de verdad de la compuerta {nombre} en codificacion {codificacion}.",
    )


# ---------------------------------------------------------------------------
# Reconocimiento de las letras X y O  (ICE-claseRN03.md)
# ---------------------------------------------------------------------------

#: Retinas de 5x5 pixeles.  '#' es pixel encendido (+1), '.' apagado (-1).
_PLANTILLAS_LETRAS = {
    "X": [
        "#...#",
        ".#.#.",
        "..#..",
        ".#.#.",
        "#...#",
    ],
    "O": [
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ],
}


def _plantilla_a_vector(filas: list[str]) -> np.ndarray:
    """Convierte una retina dibujada con '#' y '.' en un vector bipolar."""
    return np.array([1.0 if c == "#" else -1.0 for fila in filas for c in fila])


def letra(nombre: str) -> np.ndarray:
    """Devuelve la retina bipolar (25,) de la letra 'X' u 'O'."""
    return _plantilla_a_vector(_PLANTILLAS_LETRAS[nombre.upper()])


def letras_xo(con_ruido: int = 0, prob_ruido: float = 0.1, semilla: int = 0) -> Conjunto:
    """Conjunto de entrenamiento del clasificador de letras X / O.

    Reproduce la figura "Funcionamiento de un Perceptron" de
    `ICE-claseRN03.md`: la red tiene m0 = 25 entradas (una retina de 5x5) y
    m1 = 2 neuronas de salida.  La primera neurona responde +1 ante una X y -1
    en caso contrario; la segunda hace lo propio con la O.

    Parameters
    ----------
    con_ruido:
        Numero de copias ruidosas *adicionales* que se generan por letra.  Con 0
        el conjunto son los dos prototipos limpios.
    prob_ruido:
        Probabilidad de invertir cada pixel en las copias ruidosas.
    semilla:
        Semilla del generador para que el experimento sea reproducible.
    """
    rng = np.random.default_rng(semilla)
    patrones: list[np.ndarray] = []
    objetivos: list[list[float]] = []

    for indice, nombre in enumerate(("X", "O")):
        base = letra(nombre)
        objetivo = [-1.0, -1.0]
        objetivo[indice] = 1.0
        patrones.append(base)
        objetivos.append(list(objetivo))
        for _ in range(con_ruido):
            mascara = rng.random(base.size) < prob_ruido
            copia = base.copy()
            copia[mascara] *= -1.0
            patrones.append(copia)
            objetivos.append(list(objetivo))

    return Conjunto(
        X=np.array(patrones),
        d=np.array(objetivos),
        nombre="Letras X/O (retina 5x5)",
        nombres_entradas=[f"p{i}" for i in range(25)],
        nombres_salidas=["neurona_X", "neurona_O"],
        descripcion=(
            "Retinas bipolares de 5x5 pixeles. Dos neuronas de salida, una por clase, "
            "tal como en la figura 'Funcionamiento de un Perceptron' de ICE-claseRN03."
        ),
    )


def contaminar(patron: np.ndarray, n_pixeles: int, semilla: int = 0) -> np.ndarray:
    """Invierte exactamente `n_pixeles` posiciones elegidas al azar.

    Sirve para comprobar la tolerancia a "estimulos contaminados" que
    `claseRN01.md` atribuye a las redes neuronales: "es capaz de recuperar
    informacion a partir de estimulos incompletos, ruidosos o parcialmente
    erroneos".
    """
    rng = np.random.default_rng(semilla)
    copia = np.array(patron, dtype=float).copy()
    indices = rng.choice(copia.size, size=n_pixeles, replace=False)
    copia[indices] *= -1.0
    return copia


# ---------------------------------------------------------------------------
# Descodificador binario -> decimal  (ICE-claseRN04.md)
# ---------------------------------------------------------------------------

def decodificador_binario(n_bits: int = 3, bit_mas_significativo_primero: bool = True) -> Conjunto:
    """Los 2^n patrones binarios de `n_bits` con su valor decimal como objetivo.

    `ICE-claseRN04.md` propone este problema como ejemplo del ADALINE: "el
    descodificador recibe una entrada en binario y produce como salida su valor
    en decimal".  La solucion analitica existe y es lineal --- el valor decimal
    es una suma ponderada de los bits con pesos 4, 2, 1 para tres bits --- de
    modo que el ADALINE deberia poder aprenderla *exactamente* (error cuadratico
    medio -> 0), lo que convierte al problema en un banco de pruebas ideal para
    verificar la regla Delta.

    Nota sobre el enunciado: la formula que aparece en la clase, d = sum 2^i xi
    con i de 1 a n, produce pesos 2, 4, 8; aqui se usa la convencion estandar
    d = sum 2^(i-1) xi (pesos 4, 2, 1 para tres bits), que es la que hace que
    el rango de salida sea 0..2^n - 1.  El ADALINE aprende los pesos que
    correspondan a los datos, asi que el cambio de convencion solo altera la
    escala de los pesos optimos, no la naturaleza del problema.
    """
    if n_bits < 1:
        raise ValueError("n_bits debe ser >= 1")

    filas = []
    for valor in range(2 ** n_bits):
        bits = [(valor >> k) & 1 for k in range(n_bits)]
        if bit_mas_significativo_primero:
            bits = bits[::-1]
        filas.append(bits)

    X = np.array(filas, dtype=float)
    d = np.arange(2 ** n_bits, dtype=float)

    if bit_mas_significativo_primero:
        nombres = [f"b{n_bits - 1 - i}" for i in range(n_bits)]
    else:
        nombres = [f"b{i}" for i in range(n_bits)]

    return Conjunto(
        X=X,
        d=d,
        nombre=f"Descodificador binario->decimal ({n_bits} bits)",
        nombres_entradas=nombres,
        nombres_salidas=["decimal"],
        descripcion=(
            "Los 2^n patrones binarios de n bits etiquetados con su valor decimal. "
            "Problema linealmente resoluble: pesos optimos 2^(n-1), ..., 2, 1 y sesgo 0."
        ),
    )


def pesos_optimos_decodificador(n_bits: int = 3, bit_mas_significativo_primero: bool = True):
    """Solucion analitica del descodificador: (w0, w1, ..., wn) con w0 el sesgo."""
    potencias = np.array([2.0 ** k for k in range(n_bits)])
    if bit_mas_significativo_primero:
        potencias = potencias[::-1]
    return np.concatenate(([0.0], potencias))


# ---------------------------------------------------------------------------
# Nubes de puntos sinteticas
# ---------------------------------------------------------------------------

def nubes_separables(
    n_por_clase: int = 40,
    separacion: float = 3.0,
    dispersion: float = 0.8,
    semilla: int = 0,
    bipolar: bool = True,
) -> Conjunto:
    """Dos nubes gaussianas linealmente separables en el plano.

    Se usa para ilustrar la figura "Algoritmo Perceptronico: Infinitas
    Soluciones" de `ICE-claseRN03.md`: cuando un problema es linealmente
    separable no hay una solucion, sino infinitas.
    """
    rng = np.random.default_rng(semilla)
    centro = np.array([separacion / 2.0, separacion / 2.0])
    clase_pos = rng.normal(loc=centro, scale=dispersion, size=(n_por_clase, 2))
    clase_neg = rng.normal(loc=-centro, scale=dispersion, size=(n_por_clase, 2))

    X = np.vstack([clase_pos, clase_neg])
    etiqueta_neg = -1.0 if bipolar else 0.0
    d = np.concatenate([np.ones(n_por_clase), np.full(n_por_clase, etiqueta_neg)])

    return Conjunto(
        X=X,
        d=d,
        nombre="Nubes gaussianas separables",
        nombres_entradas=["x1", "x2"],
        nombres_salidas=["clase"],
        descripcion="Dos nubes gaussianas separables por una recta; infinitas soluciones.",
    )
