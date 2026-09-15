"""
Regla de Hebb (1949) para redes unicapa.

Referencia: `clases/ICE-claseRN03.md`, seccion "Regla de Hebb".

"Si dos neuronas a ambos lados de la sinapsis estaban activas (o inactivas)
simultaneamente, entonces las sinapsis entre ellas se reforzaban, y si se
activaban (o desactivaban) asincronicamente, se debilitaban":

    Delta w_ji = a * y_j(n) * x_i(n),     a > 0

En el aprendizaje *supervisado* de una red unicapa se usa la salida deseada
d_j(n) en lugar de la salida producida y_j(n), de modo que el incremento es

    Delta w_ji = a * d_j(n) * x_i(n)

Se trata de un aprendizaje **de una sola pasada**: basta presentar cada patron
una vez, sin iterar, sin comprobar errores y sin criterio de parada.  Esto lo
hace extremadamente barato, pero tambien mucho mas debil que la regla
perceptronica: solo funciona cuando los patrones estan suficientemente
"descorrelacionados".

Por que la codificacion importa
-------------------------------
Con codificacion binaria {0, 1}, un patron con x_i = 0 no aporta nada al peso
w_ji (el incremento es proporcional a x_i), de modo que los patrones "apagados"
son invisibles para el aprendizaje: la regla de Hebb **no** resuelve el AND
binario.  Con codificacion bipolar {-1, +1} todas las entradas contribuyen y la
regla si resuelve el AND y el OR.  Los experimentos de este repositorio
verifican numericamente ambas afirmaciones.
"""

from __future__ import annotations

import numpy as np

from .activaciones import EscalonBipolar, FuncionActivacion


class RedHebb:
    """Red unicapa entrenada con la regla de Hebb supervisada.

    Parameters
    ----------
    n_entradas, n_salidas:
        Dimensiones m0 y m1 de la capa.
    razon_aprendizaje:
        La constante "a" de la clase, llamada alli razon de aprendizaje.  Como
        el aprendizaje es de una sola pasada y la frontera de decision no
        cambia al escalar todos los pesos, su valor solo afecta la escala de
        los pesos, no la clasificacion.
    activacion:
        Por omision el escalon bipolar.
    """

    def __init__(
        self,
        n_entradas: int,
        n_salidas: int = 1,
        razon_aprendizaje: float = 1.0,
        activacion: FuncionActivacion | None = None,
    ):
        self.n_entradas = int(n_entradas)
        self.n_salidas = int(n_salidas)
        self.a = float(razon_aprendizaje)
        self.phi = activacion or EscalonBipolar()
        self.W = np.zeros((self.n_salidas, self.n_entradas + 1))
        self.trayectoria_pesos: list[np.ndarray] = [self.W.copy()]

    @staticmethod
    def _con_sesgo(X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def entrenar(self, X, d, verbose: bool = False) -> "RedHebb":
        """Una sola pasada: para cada patron, W += a * d^T x."""
        Xe = self._con_sesgo(X)
        d = np.asarray(d, dtype=float)
        if d.ndim == 1:
            d = d.reshape(-1, 1)

        self.W = np.zeros((self.n_salidas, self.n_entradas + 1))
        self.trayectoria_pesos = [self.W.copy()]

        for n in range(Xe.shape[0]):
            self.W += self.a * np.outer(d[n], Xe[n])
            self.trayectoria_pesos.append(self.W.copy())
            if verbose:
                print(f"  patron {n + 1}: W = {np.round(self.W, 3).tolist()}")
        return self

    def predecir(self, X) -> np.ndarray:
        y = self.phi(self._con_sesgo(X) @ self.W.T)
        return y[:, 0] if self.n_salidas == 1 else y

    __call__ = predecir

    def exactitud(self, X, d) -> float:
        d = np.asarray(d, dtype=float)
        y = self.predecir(X)
        if d.ndim == 1:
            return float(np.mean(y == d))
        return float(np.mean(np.all(y == d, axis=1)))

    def recta_frontera(self, neurona: int = 0) -> tuple[float, float, float]:
        if self.n_entradas != 2:
            raise ValueError("la recta frontera solo esta definida para 2 entradas")
        w = self.W[neurona]
        return float(w[0]), float(w[1]), float(w[2])

    def resumen(self) -> str:
        pesos = "\n".join(
            f"  neurona {j}: " + "  ".join(f"{v:7.3f}" for v in self.W[j])
            for j in range(self.n_salidas)
        )
        return (
            f"Regla de Hebb  m0={self.n_entradas}  m1={self.n_salidas}  a={self.a}\n"
            f"Pesos tras una sola pasada (columna 0 = sesgo):\n{pesos}"
        )
