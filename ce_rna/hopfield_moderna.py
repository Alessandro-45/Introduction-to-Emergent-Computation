"""
Red de Hopfield moderna (continua): la memoria asociativa que es una capa de atencion.

Referencia: Ramsauer, H. et al. (2021). *Hopfield Networks is All You Need*.
ICLR 2021.  arXiv:2008.02217.  Guia de evaluacion, Actividad 6 (seminario).

De Hopfield (1982) a Hopfield moderno
-------------------------------------
La red clasica (`ce_rna.hopfield`) guarda P patrones en una matriz de pesos
W = (1/N) sum_mu xi^mu xi^mu^T y su energia es cuadratica en el estado:

    E(s) = -(1/2) s^T W s = -(1/(2N)) sum_mu (xi^mu . s)^2

Cada patron "tira" del estado con una fuerza proporcional al **cuadrado** de su
solapamiento.  Con tan poca separacion entre el patron correcto y los demas la
diafonia crece rapido y la capacidad se queda en ~0.14 N patrones.

Krotov y Hopfield (2016) cambiaron el cuadrado por una potencia F(x) = x^n, y
Demircigil et al. (2017) por la exponencial F(x) = exp(x): la contribucion del
patron mas parecido domina a las demas de forma exponencial y la capacidad
pasa a crecer exponencialmente con la dimension.  Ramsauer et al. (2021) llevan
esa idea a estados **continuos** con la energia

    E(xi) = -lse(beta, X^T xi) + (1/2) xi^T xi + (1/beta) log P + (1/2) M^2

donde X = (x_1, ..., x_P) son los patrones almacenados (columnas, d x P),
lse(beta, z) = (1/beta) log sum_mu exp(beta z_mu) es el "log-sum-exp" (un maximo
suave) y M = max_mu ||x_mu||.  Los dos ultimos terminos son constantes y solo
sirven para que 0 <= E <= 2 M^2.

Regla de actualizacion
----------------------
Minimizando E con el procedimiento concavo-convexo (CCCP) se obtiene

    xi_nuevo = X softmax(beta X^T xi)

1. se compara el estado con cada patron (producto escalar X^T xi),
2. softmax convierte esas similitudes en pesos p_mu >= 0 que suman 1,
3. el nuevo estado es el promedio ponderado de los patrones.

El paper demuestra que la energia nunca aumenta con esta regla, que la
iteracion converge a un punto estacionario y que, si los patrones estan bien
separados, **una sola actualizacion** basta para recuperar el patron con un
error exponencialmente pequeno.

beta y los tres tipos de punto fijo
-----------------------------------
beta es una temperatura inversa.  Con beta pequeno softmax reparte el peso entre
todos los patrones y el punto fijo es (casi) la media de todos: un **punto fijo
global**.  Con beta intermedio el peso se reparte entre un grupo de patrones
parecidos: un **estado metaestable**.  Con beta grande todo el peso va al
patron mas parecido: un punto fijo **cerca de un unico patron**, que es la
memoria asociativa clasica.

Equivalencia con la atencion de los Transformers
------------------------------------------------
Si se aplican proyecciones aprendidas Q = R W_Q, K = Y W_K, V = K W_V (= Y W_K W_V)
y beta = 1 / sqrt(d_k), la regla de actualizacion aplicada a varios estados a
la vez es exactamente

    Z = softmax(beta Q K^T) V

la atencion de producto escalar de Vaswani et al. (2017).  Una capa de
atencion es, por tanto, un paso de recuperacion de una red de Hopfield moderna.

Convencion de este modulo
-------------------------
Como en `ce_rna.hopfield`, los patrones se pasan **por filas** (matriz P x d);
internamente las formulas del paper usan X = patrones^T.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def softmax(z, beta: float = 1.0) -> np.ndarray:
    """softmax(beta z) a lo largo del ultimo eje, estable para argumentos grandes."""
    z = beta * np.asarray(z, dtype=float)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def lse(z, beta: float = 1.0) -> float:
    """lse(beta, z) = (1/beta) log sum exp(beta z), calculado sin desbordamiento.

    Es un maximo suave: max(z) <= lse(beta, z) <= max(z) + log(len(z)) / beta.
    """
    z = np.asarray(z, dtype=float)
    m = np.max(z)
    return float(m + np.log(np.sum(np.exp(beta * (z - m)))) / beta)


def atencion(Q, K, V, beta: float | None = None) -> np.ndarray:
    """Atencion de producto escalar: softmax(beta Q K^T) V (filas = elementos).

    Por omision beta = 1 / sqrt(d_k), como en los Transformers.
    """
    Q, K, V = (np.atleast_2d(np.asarray(M, dtype=float)) for M in (Q, K, V))
    if beta is None:
        beta = 1.0 / np.sqrt(K.shape[1])
    return softmax(Q @ K.T, beta) @ V


# ---------------------------------------------------------------------------
# Red de Hopfield moderna
# ---------------------------------------------------------------------------

@dataclass
class ResultadoModerno:
    """Resultado de una recuperacion con la red moderna."""

    estado: np.ndarray
    iteraciones: int                    # actualizaciones aplicadas
    convergio: bool
    energia: list[float] = field(default_factory=list)
    pesos: np.ndarray | None = None     # softmax(beta X^T xi) en el estado final


class RedHopfieldModerna:
    """Red de Hopfield moderna con estados continuos (Ramsauer et al., 2021).

    Parameters
    ----------
    beta:
        Temperatura inversa de softmax.  Con patrones bipolares de dimension d
        los productos escalares van de -d a d, y beta del orden de 1/d ya
        distingue patrones que difieren en pocos pixeles.
    """

    def __init__(self, beta: float = 1.0):
        if beta <= 0:
            raise ValueError("beta debe ser positivo")
        self.beta = float(beta)
        self.patrones = np.zeros((0, 0))

    # -- almacenamiento ------------------------------------------------------
    def almacenar(self, patrones) -> "RedHopfieldModerna":
        """Guarda los patrones tal cual: no hay matriz de pesos ni entrenamiento.

        La "memoria" es la propia matriz X; por eso anadir un patron no deforma
        a los demas, a diferencia de la suma de productos exteriores de Hebb.
        """
        P = np.atleast_2d(np.asarray(patrones, dtype=float))
        if P.shape[0] == 0:
            raise ValueError("hace falta al menos un patron")
        self.patrones = P.copy()
        return self

    @property
    def d(self) -> int:
        return self.patrones.shape[1]

    @property
    def P(self) -> int:
        return self.patrones.shape[0]

    # -- dinamica ------------------------------------------------------------
    def similitudes(self, xi) -> np.ndarray:
        """X^T xi: producto escalar del estado con cada patron."""
        return self.patrones @ np.asarray(xi, dtype=float)

    def pesos(self, xi) -> np.ndarray:
        """p = softmax(beta X^T xi): cuanto contribuye cada patron al nuevo estado."""
        return softmax(self.similitudes(xi), self.beta)

    def actualizar(self, xi) -> np.ndarray:
        """Una actualizacion: xi_nuevo = X softmax(beta X^T xi)."""
        return self.pesos(xi) @ self.patrones

    def energia(self, xi) -> float:
        """E = -lse(beta, X^T xi) + xi.xi/2 + log(P)/beta + M^2/2."""
        xi = np.asarray(xi, dtype=float)
        M = np.max(np.linalg.norm(self.patrones, axis=1))
        return (-lse(self.similitudes(xi), self.beta) + 0.5 * xi @ xi
                + np.log(self.P) / self.beta + 0.5 * M ** 2)

    def recuperar(self, xi0, max_iteraciones: int = 100, tolerancia: float = 1e-10) -> ResultadoModerno:
        """Aplica la regla de actualizacion hasta que el estado deja de cambiar.

        Se considera convergido cuando ||xi_nuevo - xi|| <= tolerancia * sqrt(d).
        Todas las neuronas se actualizan a la vez (como en una capa de atencion):
        no hace falta dinamica asincrona para que la energia baje.
        """
        xi = np.asarray(xi0, dtype=float).copy()
        energias = [self.energia(xi)]
        umbral = tolerancia * np.sqrt(xi.size)
        for it in range(1, max_iteraciones + 1):
            nuevo = self.actualizar(xi)
            cambio = np.linalg.norm(nuevo - xi)
            xi = nuevo
            energias.append(self.energia(xi))
            if cambio <= umbral:
                return ResultadoModerno(xi, it, True, energias, self.pesos(xi))
        return ResultadoModerno(xi, max_iteraciones, False, energias, self.pesos(xi))

    # -- interpretacion ------------------------------------------------------
    def separacion(self) -> np.ndarray:
        """Delta_mu = x_mu.x_mu - max_{nu != mu} x_mu.x_nu (separacion de cada patron).

        Es la magnitud que controla el teorema de recuperacion del paper: cuanto
        mayor sea beta * Delta_mu, mas rapido y exacto se recupera x_mu.
        """
        G = self.patrones @ self.patrones.T
        diag = np.diag(G).copy()
        np.fill_diagonal(G, -np.inf)
        return diag - G.max(axis=1)

    @staticmethod
    def patrones_efectivos(pesos) -> float:
        """Numero efectivo de patrones que se promedian: 1 / sum p_mu^2.

        Vale 1 si todo el peso esta en un patron y P si se reparte por igual.
        """
        p = np.asarray(pesos, dtype=float)
        return float(1.0 / np.sum(p ** 2))

    def tipo_punto_fijo(self, pesos, umbral_unico: float = 0.9) -> str:
        """Clasifica un punto fijo por como reparte softmax el peso.

        ``"unico"`` si un patron se lleva al menos `umbral_unico` del peso,
        ``"global"`` si se promedian (casi) todos los patrones, y
        ``"metaestable"`` en el caso intermedio.
        """
        p = np.asarray(pesos, dtype=float)
        if p.max() >= umbral_unico:
            return "unico"
        if self.patrones_efectivos(p) >= 0.5 * self.P:
            return "global"
        return "metaestable"

    def identificar(self, estado) -> tuple[int, str]:
        """Compara sgn(estado) con los patrones almacenados (bipolares).

        Devuelve ``(mu, "patron")`` si coincide exactamente con x_mu y
        ``(-1, "otro")`` en caso contrario.
        """
        s = np.where(np.asarray(estado) >= 0, 1.0, -1.0)
        iguales = np.flatnonzero(np.all(self.patrones == s, axis=1))
        if iguales.size:
            return int(iguales[0]), "patron"
        return -1, "otro"

    def resumen(self) -> str:
        return (f"Hopfield moderna | d = {self.d} | {self.P} patrones | beta = {self.beta:g} | "
                f"separacion minima Delta = {self.separacion().min():g}")
