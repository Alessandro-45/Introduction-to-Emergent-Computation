"""
Modelo de McCulloch-Pitts (1943) y compuertas logicas AND / OR.

Referencia: `clases/claseRN02.md`, seccion "El Comienzo: McCulloch-Pitts".

Caracteristicas del modelo, literalmente como se enuncian en la clase:

1. Las neuronas son del tipo **binario**.
2. Los umbrales y las sinapsis **se mantienen fijas** (no hay aprendizaje).
3. La funcion de activacion es del tipo **escalon**.

Con estas tres reglas, McCulloch y Pitts demostraron que "todas las funciones
logicas se pueden describir mediante combinaciones apropiadas de neuronas de
este tipo, y que por lo tanto se podia crear, en principio, una red capaz de
resolver cualquier funcion computable".

Este modulo implementa:

* `NeuronaMCP`   : la unidad de proceso (suma ponderada + umbral + escalon).
* `RedMCP`       : un grafo acilico de neuronas MCP que permite *componer*
                   compuertas (por ejemplo, construir el XOR a partir de
                   AND, OR y NOT), que es justamente el argumento de
                   universalidad del articulo original.
* Constructores  : `neurona_and`, `neurona_or`, `neurona_not`, `neurona_nand`,
                   `neurona_nor`, `red_xor`.

No hay entrenamiento en este modulo: los pesos se *disenan* a mano.  El
aprendizaje llega con el perceptron (`ce_rna.perceptron`) y con el ADALINE
(`ce_rna.adaline`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activaciones import Escalon


@dataclass
class NeuronaMCP:
    """Neurona binaria de McCulloch-Pitts.

    El potencial de la neurona es la suma ponderada de sus entradas

        y^(in) = sum_i  w_i * x_i

    y su salida aplica la "ley de todo o nada" del impulso nervioso descrita en
    `claseRN01.md`:

        y = 1  si  y^(in) >= umbral
        y = 0  si  y^(in) <  umbral

    Attributes
    ----------
    pesos:
        Vector de pesos sinapticos w_i.  Positivos = conexiones excitadoras;
        negativos = conexiones inhibidoras (`claseRN01.md`: "de acuerdo con el
        signo del peso w_kj se tienen conexiones excitadoras cuando es positivo,
        y conexiones inhibidoras cuando es negativo").
    umbral:
        Nivel a partir del cual la neurona dispara (theta).
    nombre:
        Etiqueta usada en las figuras y en los reportes.
    """

    pesos: np.ndarray
    umbral: float
    nombre: str = "MCP"

    def __post_init__(self) -> None:
        self.pesos = np.asarray(self.pesos, dtype=float)
        if self.pesos.ndim != 1:
            raise ValueError("los pesos de una neurona MCP deben ser un vector 1-D")
        self.umbral = float(self.umbral)
        self._phi = Escalon(umbral=self.umbral)

    # -- propiedades -------------------------------------------------------
    @property
    def n_entradas(self) -> int:
        return self.pesos.size

    # -- calculo -----------------------------------------------------------
    def potencial(self, X) -> np.ndarray:
        """Suma ponderada y^(in) para uno o varios patrones."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_entradas:
            raise ValueError(
                f"la neurona '{self.nombre}' espera {self.n_entradas} entradas, "
                f"recibio {X.shape[1]}"
            )
        return X @ self.pesos

    def activar(self, X) -> np.ndarray:
        """Salida binaria {0, 1} de la neurona."""
        return self._phi(self.potencial(X))

    __call__ = activar

    # -- utilidades --------------------------------------------------------
    def tabla_verdad(self):
        """Tabla de verdad completa de la neurona (solo entradas binarias)."""
        import pandas as pd

        n = self.n_entradas
        combinaciones = np.array(
            [[(v >> (n - 1 - k)) & 1 for k in range(n)] for v in range(2 ** n)],
            dtype=float,
        )
        potenciales = self.potencial(combinaciones)
        salidas = self._phi(potenciales)
        datos = {f"x{i + 1}": combinaciones[:, i] for i in range(n)}
        datos["y_in"] = potenciales
        datos["y"] = salidas
        return pd.DataFrame(datos)

    def describir(self) -> str:
        terminos = " + ".join(
            f"{w:g}*x{i + 1}" for i, w in enumerate(self.pesos)
        )
        return f"{self.nombre}: y = 1 si ({terminos}) >= {self.umbral:g}, si no y = 0"


# ---------------------------------------------------------------------------
# Constructores de compuertas elementales
# ---------------------------------------------------------------------------

def neurona_and(n_entradas: int = 2) -> NeuronaMCP:
    """Compuerta AND de `n_entradas` entradas.

    Para dos entradas reproduce exactamente el ejemplo de `claseRN02.md`:
    pesos (1, 1) y umbral 2.  La neurona solo dispara cuando *todas* las
    entradas valen 1, porque ese es el unico caso en que la suma alcanza el
    umbral n.
    """
    return NeuronaMCP(pesos=np.ones(n_entradas), umbral=float(n_entradas), nombre="AND")


def neurona_or(n_entradas: int = 2) -> NeuronaMCP:
    """Compuerta OR de `n_entradas` entradas.

    `claseRN02.md` la presenta con pesos (2, 2) y umbral 2: basta con que una
    entrada valga 1 para alcanzar el umbral.  Se conserva esa parametrizacion
    original (equivalente a pesos 1 y umbral 1, que seria la forma canonica).
    """
    return NeuronaMCP(pesos=np.full(n_entradas, 2.0), umbral=2.0, nombre="OR")


def neurona_not() -> NeuronaMCP:
    """Compuerta NOT: una sola conexion inhibidora.

    Con peso -1 y umbral 0: y^(in) = -x, de modo que x = 0 -> y = 1 y
    x = 1 -> y = 0.
    """
    return NeuronaMCP(pesos=np.array([-1.0]), umbral=0.0, nombre="NOT")


def neurona_nand(n_entradas: int = 2) -> NeuronaMCP:
    """NAND: pesos -1 y umbral -(n-1).  Dispara salvo cuando todas valen 1."""
    return NeuronaMCP(
        pesos=np.full(n_entradas, -1.0),
        umbral=-(n_entradas - 1.0),
        nombre="NAND",
    )


def neurona_nor(n_entradas: int = 2) -> NeuronaMCP:
    """NOR: pesos -1 y umbral 0.  Solo dispara cuando todas las entradas son 0."""
    return NeuronaMCP(pesos=np.full(n_entradas, -1.0), umbral=0.0, nombre="NOR")


# ---------------------------------------------------------------------------
# Redes de neuronas MCP
# ---------------------------------------------------------------------------

class RedMCP:
    """Red acilica (feed-forward) de neuronas de McCulloch-Pitts.

    Cada nodo se declara con el nombre de sus fuentes, que pueden ser entradas
    externas o nodos declarados previamente.  Como las neuronas se anaden en
    orden topologico, la evaluacion es un simple recorrido secuencial.

    Ejemplo
    -------
    >>> red = RedMCP(["x1", "x2"])
    >>> _ = red.agregar("h_or", neurona_or(2), ["x1", "x2"])
    >>> _ = red.agregar("h_and", neurona_and(2), ["x1", "x2"])
    >>> _ = red.agregar("y", NeuronaMCP([1.0, -2.0], 1.0, "XOR"), ["h_or", "h_and"])
    >>> red.evaluar([[0, 1]])["y"]
    array([1.])
    """

    def __init__(self, entradas: list[str], nombre: str = "RedMCP"):
        self.entradas = list(entradas)
        self.nombre = nombre
        self.nodos: list[tuple[str, NeuronaMCP, list[str]]] = []

    def agregar(self, nombre: str, neurona: NeuronaMCP, fuentes: list[str]) -> "RedMCP":
        """Anade una neurona conectada a `fuentes`.  Devuelve la red (encadenable)."""
        disponibles = set(self.entradas) | {n for n, _, _ in self.nodos}
        faltantes = [f for f in fuentes if f not in disponibles]
        if faltantes:
            raise KeyError(f"fuentes no declaradas: {faltantes}")
        if len(fuentes) != neurona.n_entradas:
            raise ValueError(
                f"'{nombre}' recibe {len(fuentes)} fuentes pero la neurona "
                f"espera {neurona.n_entradas} entradas"
            )
        self.nodos.append((nombre, neurona, list(fuentes)))
        return self

    @property
    def nombres_salida(self) -> list[str]:
        return [n for n, _, _ in self.nodos]

    def evaluar(self, X) -> dict[str, np.ndarray]:
        """Propaga los patrones y devuelve la activacion de *todos* los nodos."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.entradas):
            raise ValueError(
                f"la red espera {len(self.entradas)} entradas, recibio {X.shape[1]}"
            )
        senales: dict[str, np.ndarray] = {
            nombre: X[:, i] for i, nombre in enumerate(self.entradas)
        }
        for nombre, neurona, fuentes in self.nodos:
            entrada = np.column_stack([senales[f] for f in fuentes])
            senales[nombre] = neurona.activar(entrada)
        return senales

    def salida(self, X) -> np.ndarray:
        """Activacion del ultimo nodo declarado (la salida de la red)."""
        if not self.nodos:
            raise RuntimeError("la red no tiene neuronas")
        return self.evaluar(X)[self.nodos[-1][0]]

    def tabla_verdad(self):
        """Tabla con las entradas, todas las activaciones internas y la salida."""
        import pandas as pd

        n = len(self.entradas)
        combinaciones = np.array(
            [[(v >> (n - 1 - k)) & 1 for k in range(n)] for v in range(2 ** n)],
            dtype=float,
        )
        senales = self.evaluar(combinaciones)
        datos = {nombre: senales[nombre] for nombre in self.entradas}
        for nombre, _, _ in self.nodos:
            datos[nombre] = senales[nombre]
        return pd.DataFrame(datos)


def red_xor() -> RedMCP:
    """Red de tres neuronas MCP que calcula el XOR.

    El XOR **no** es linealmente separable (`ICE-claseRN03.md`: "es imposible
    encontrar una linea recta que deje a un lado las entradas que deben producir
    0, y al otro, las que deben producir 1"), de modo que ninguna neurona MCP
    aislada puede calcularlo.  Sin embargo si puede construirse componiendo
    neuronas, que es precisamente el resultado de universalidad de
    McCulloch-Pitts:

        XOR(x1, x2) = OR(x1, x2)  AND  NOT( AND(x1, x2) )

    La neurona de salida implementa ese "AND con una entrada inhibidora" con
    pesos (+1, -2) y umbral 1: la conexion inhibidora procedente de h_and es lo
    bastante fuerte como para cancelar la excitacion de h_or.
    """
    red = RedMCP(["x1", "x2"], nombre="XOR (composicion de neuronas MCP)")
    red.agregar("h_or", neurona_or(2), ["x1", "x2"])
    red.agregar("h_and", neurona_and(2), ["x1", "x2"])
    red.agregar("y", NeuronaMCP(pesos=[1.0, -2.0], umbral=1.0, nombre="XOR"), ["h_or", "h_and"])
    return red
