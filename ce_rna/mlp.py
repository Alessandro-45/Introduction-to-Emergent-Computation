"""
Perceptron Multicapa (MLP) y algoritmo de Retropropagacion (Backpropagation).

Referencia teorica
------------------
* `clases/claseRN01.md`, seccion "Algunas funciones de activacion".
* `clases/ICE-claseRN03.md`, secciones "Red Multicapa", "Convexidad" y "Paralelismo".
* Rumelhart, Hinton y Williams (1986): "Learning representations by back-propagating errors".

Fundamentos matematicos
-----------------------
Una red feedforward de L capas (capa 0 = entradas, capas 1 a L = capas neuronales).
Para un patron de entrada x:

1. Propagacion hacia adelante (Forward):
   Para cada capa l = 1, ..., L:
       v^(l) = W^(l) * a^(l-1) + b^(l)           (potencial post-sinaptico)
       a^(l) = phi^(l)( v^(l) )                   (activacion / salida)
   donde a^(0) = x.

2. Error cuadratico medio (ECM):
   Para un patron p con objetivo d^p y salida a^(L):
       E^p = 1/2 * sum_{k=1}^{m_L} (d_k^p - a_k^(L))^2
   Sobre N patrones:
       E = 1/N * sum_{p=1}^N E^p

3. Retropropagacion del error (Backpropagation):
   Por regla de la cadena:
   - Capa de salida L:
       delta_k^(L) = (d_k - a_k^(L)) * (phi^(L))'(v_k^(L))
   - Capas ocultas l = L-1, ..., 1:
       delta_j^(l) = [ sum_k delta_k^(l+1) * W_{kj}^(l+1) ] * (phi^(l))'(v_j^(l))

4. Actualizacion de pesos (descenso por gradiente con momento opcional):
   - Con la definicion de delta anterior, -dE/dW_{ji}^(l) = delta_j^(l) * a_i^(l-1),
     por lo que los pesos se incrementan sumando el termino:
       Delta W_{ji}^(l)(t) = eta * delta_j^(l) * a_i^(l-1) + alpha * Delta W_{ji}^(l)(t-1)
       Delta b_j^(l)(t)    = eta * delta_j^(l)             + alpha * Delta b_j^(l)(t-1)
       W^(l) <- W^(l) + Delta W^(l)
       b^(l) <- b^(l) + Delta b^(l)
   donde eta > 0 es la razon de aprendizaje y alpha in [0, 1) es el coeficiente de momento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .activaciones import FuncionActivacion, Identidad, Sigmoide, obtener


@dataclass
class HistorialEpocaMLP:
    """Instantanea del entrenamiento al terminar una epoca."""

    epoca: int
    ecm_entrenamiento: float
    ecm_validacion: float | None = None


class PerceptronMulticapa:
    """Perceptron Multicapa entrenado mediante Retropropagacion (Backpropagation).

    Parameters
    ----------
    arquitectura:
        Secuencia de enteros con el numero de unidades por capa:
        `(m_0, m_1, ..., m_L)`. Por ejemplo, `(2, 2, 1)` para XOR.
    activacion_oculta:
        Instancia de `FuncionActivacion` o nombre en el catalogo para las capas
        ocultas 1 a L-1. Por omision es "sigmoide".
    activacion_salida:
        Instancia de `FuncionActivacion` o nombre en el catalogo para la capa de
        salida L. Por omision es "sigmoide" (clasificacion binaria o probabilidad).
        Para aproximacion de funciones continuas suele utilizarse "identidad".
    razon_aprendizaje:
        eta, razon de aprendizaje para el descenso por el gradiente (0 < eta).
    momento:
        alpha in [0, 1), coeficiente de momento que acelera en mesetas y frena
        oscilaciones.
    modo:
        ``"estocastico"`` actualiza pesos patron a patron; ``"lote"`` acumula el
        gradiente de todo el conjunto y actualiza una vez por epoca de forma vectorizada.
    tolerancia:
        Criterio de parada por error: si el ECM cae por debajo, se detiene.
    max_epocas:
        Criterio de parada por iteraciones maximas.
    semilla:
        Semilla para el generador aleatorio de NumPy (reproducibilidad).
    """

    def __init__(
        self,
        arquitectura: Sequence[int],
        activacion_oculta: FuncionActivacion | str = "sigmoide",
        activacion_salida: FuncionActivacion | str = "sigmoide",
        razon_aprendizaje: float = 0.5,
        momento: float = 0.0,
        modo: str = "estocastico",
        tolerancia: float = 1e-4,
        max_epocas: int = 5000,
        semilla: int | None = None,
    ):
        if len(arquitectura) < 2:
            raise ValueError("La arquitectura debe tener al menos 2 capas (entrada y salida)")
        if any(n <= 0 for n in arquitectura):
            raise ValueError("Cada capa debe tener al menos 1 neurona")
        if razon_aprendizaje <= 0:
            raise ValueError("razon_aprendizaje debe ser positiva")
        if not (0.0 <= momento < 1.0):
            raise ValueError("momento debe estar en el intervalo [0, 1)")
        if modo not in ("estocastico", "lote"):
            raise ValueError("modo debe ser 'estocastico' o 'lote'")

        self.arquitectura = tuple(arquitectura)
        self.n_capas = len(self.arquitectura) - 1  # Capas con pesos (1..L)
        self.razon_aprendizaje = float(razon_aprendizaje)
        self.momento = float(momento)
        self.modo = modo
        self.tolerancia = float(tolerancia)
        self.max_epocas = int(max_epocas)
        self.semilla = semilla

        # Resolucion de activaciones
        self.act_oculta = (
            obtener(activacion_oculta) if isinstance(activacion_oculta, str) else activacion_oculta
        )
        self.act_salida = (
            obtener(activacion_salida) if isinstance(activacion_salida, str) else activacion_salida
        )

        if not self.act_oculta.derivable or not self.act_salida.derivable:
            raise ValueError("Todas las activaciones del MLP deben ser derivables para Backpropagation")

        # Inicializacion de pesos y sesgos
        self.rng = np.random.default_rng(semilla)
        self.pesos: list[np.ndarray] = []
        self.sesgos: list[np.ndarray] = []
        self.delta_pesos_prev: list[np.ndarray] = []
        self.delta_sesgos_prev: list[np.ndarray] = []
        self._inicializar_parametros()

        self.historia_error: list[float] = []
        self.historia_val: list[float] = []
        self.n_epocas_entrenadas: int = 0
        self.motivo_parada: str = "no entrenado"

    def _inicializar_parametros(self) -> None:
        """Inicializacion de pesos segun la escala de Xavier/Glorot uniforme."""
        self.pesos = []
        self.sesgos = []
        self.delta_pesos_prev = []
        self.delta_sesgos_prev = []

        for l in range(1, len(self.arquitectura)):
            fan_in = self.arquitectura[l - 1]
            fan_out = self.arquitectura[l]
            # Limite de Xavier: sqrt(6 / (fan_in + fan_out))
            limite = np.sqrt(6.0 / (fan_in + fan_out))
            W = self.rng.uniform(-limite, limite, size=(fan_out, fan_in))
            b = np.zeros(fan_out, dtype=float)

            self.pesos.append(W)
            self.sesgos.append(b)
            self.delta_pesos_prev.append(np.zeros_like(W))
            self.delta_sesgos_prev.append(np.zeros_like(b))

    def _activacion_capa(self, l: int) -> FuncionActivacion:
        """Devuelve la funcion de activacion correspondiente a la capa l (1-indexada)."""
        return self.act_salida if l == self.n_capas else self.act_oculta

    def propagar_paso(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Propagacion hacia adelante para un solo vector x (m0,).

        Devuelve:
            activaciones: [a^(0), a^(1), ..., a^(L)]
            potenciales:  [None,   v^(1), ..., v^(L)]
        """
        a = np.asarray(x, dtype=float).ravel()
        activaciones = [a]
        potenciales = [None]  # indice 0 no tiene potencial sinaptico

        for l in range(self.n_capas):
            W = self.pesos[l]
            b = self.sesgos[l]
            act = self._activacion_capa(l + 1)

            v = W @ a + b
            a = act(v)

            potenciales.append(v)
            activaciones.append(a)

        return activaciones, potenciales

    def propagar_lote(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Propagacion hacia adelante vectorizada para una matriz X (N, m0).

        Devuelve:
            activaciones: [A^(0), A^(1), ..., A^(L)] con A^(l) de forma (N, m_l)
            potenciales:  [None,   V^(1), ..., V^(L)] con V^(l) de forma (N, m_l)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        A = X
        activaciones = [A]
        potenciales = [None]

        for l in range(self.n_capas):
            W = self.pesos[l]
            b = self.sesgos[l]
            act = self._activacion_capa(l + 1)

            V = A @ W.T + b
            A = act(V)

            potenciales.append(V)
            activaciones.append(A)

        return activaciones, potenciales

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Calcula la salida continua de la red para una matriz de patrones X (N, m0)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.arquitectura[0]:
            raise ValueError(f"X tiene {X.shape[1]} entradas; se esperaban {self.arquitectura[0]}")

        acts, _ = self.propagar_lote(X)
        salidas = acts[-1]
        return salidas.ravel() if self.arquitectura[-1] == 1 and salidas.ndim == 2 else salidas

    def clasificar(self, X: np.ndarray, umbral: float = 0.5) -> np.ndarray:
        """Salida binaria {0, 1} comparando con umbral (para activacion sigmoidal)."""
        y = self.predecir(X)
        return (y >= umbral).astype(float)

    def calcular_ecm(self, X: np.ndarray, d: np.ndarray) -> float:
        """Calcula el Error Cuadratico Medio sobre (X, d) con factor 1/2 y promedio sobre N patrones."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        N = X.shape[0]
        d_mat = np.asarray(d, dtype=float).reshape(N, -1)
        y_mat = np.asarray(self.predecir(X), dtype=float).reshape(N, -1)

        error = d_mat - y_mat
        return float(np.mean(0.5 * np.sum(error ** 2, axis=1)))

    def entrenar(
        self,
        X: np.ndarray,
        d: np.ndarray,
        X_val: np.ndarray | None = None,
        d_val: np.ndarray | None = None,
    ) -> PerceptronMulticapa:
        """Entrena el perceptron multicapa mediante Backpropagation.

        Parameters
        ----------
        X:
            Matriz de entrenamiento (N, m0).
        d:
            Vector u objetivo (N,) o (N, mL).
        X_val, d_val:
            Conjunto opcional de validacion o prueba para monitorear sobreajuste.

        Returns
        -------
        self:
            La instancia entrenada.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = np.asarray(d, dtype=float)
        if X.shape[0] != d.shape[0]:
            raise ValueError("X y d deben tener la misma cantidad de filas")
        if X.shape[1] != self.arquitectura[0]:
            raise ValueError(f"X tiene {X.shape[1]} columnas; se esperaban {self.arquitectura[0]}")

        # Asegurar forma de d para multidimensionalidad uniforme
        if self.arquitectura[-1] == 1 and d.ndim == 1:
            d_formato = d.reshape(-1, 1)
        else:
            d_formato = d if d.ndim == 2 else d.reshape(-1, self.arquitectura[-1])

        N = X.shape[0]
        self.historia_error = []
        self.historia_val = []

        # Registro inicial (epoca 0)
        ecm_inicial = self.calcular_ecm(X, d)
        self.historia_error.append(ecm_inicial)
        if X_val is not None and d_val is not None:
            self.historia_val.append(self.calcular_ecm(X_val, d_val))

        if ecm_inicial <= self.tolerancia:
            self.n_epocas_entrenadas = 0
            self.motivo_parada = f"convergencia inmediata ({ecm_inicial:.2e} <= {self.tolerancia:.2e})"
            return self

        for epoca in range(1, self.max_epocas + 1):
            if self.modo == "estocastico":
                indices = self.rng.permutation(N)
                for idx in indices:
                    x_p = X[idx]
                    d_p = d_formato[idx]

                    # 1. Forward individual
                    acts, pots = self.propagar_paso(x_p)

                    # 2. Backward individual
                    deltas: list[np.ndarray] = [None] * (self.n_capas + 1)

                    # Capa de salida L
                    salida = acts[-1]
                    act_salida = self._activacion_capa(self.n_capas)
                    deriv_salida = act_salida.derivada(pots[-1])
                    deltas[self.n_capas] = (d_p - salida) * deriv_salida

                    # Capas ocultas L-1 descendiendo a 1
                    for l in range(self.n_capas - 1, 0, -1):
                        W_sig = self.pesos[l]  # pesos de capa l hacia l+1
                        delta_sig = deltas[l + 1]
                        act_oculta = self._activacion_capa(l)
                        deriv_oculta = act_oculta.derivada(pots[l])
                        propagado = W_sig.T @ delta_sig
                        deltas[l] = propagado * deriv_oculta

                    # 3. Actualizacion estocastica
                    for l in range(self.n_capas):
                        delta = deltas[l + 1]
                        a_ant = acts[l]

                        dW = self.razon_aprendizaje * np.outer(delta, a_ant) + self.momento * self.delta_pesos_prev[l]
                        db = self.razon_aprendizaje * delta + self.momento * self.delta_sesgos_prev[l]

                        self.pesos[l] += dW
                        self.sesgos[l] += db

                        self.delta_pesos_prev[l] = dW
                        self.delta_sesgos_prev[l] = db

            else:  # Modo lote (vectorizado de alto rendimiento)
                acts, pots = self.propagar_lote(X)
                deltas_lote = [None] * (self.n_capas + 1)

                # Capa de salida: (N, mL)
                deriv_salida = self._activacion_capa(self.n_capas).derivada(pots[-1])
                deltas_lote[self.n_capas] = (d_formato - acts[-1]) * deriv_salida

                # Capas ocultas L-1 descendiendo a 1
                for l in range(self.n_capas - 1, 0, -1):
                    deriv_oculta = self._activacion_capa(l).derivada(pots[l])
                    deltas_lote[l] = (deltas_lote[l + 1] @ self.pesos[l]) * deriv_oculta

                # Actualizacion en lote
                for l in range(self.n_capas):
                    grad_W = deltas_lote[l + 1].T @ acts[l]  # (m_{l+1}, m_l)
                    grad_b = np.sum(deltas_lote[l + 1], axis=0)  # (m_{l+1},)

                    dW = (self.razon_aprendizaje / N) * grad_W + self.momento * self.delta_pesos_prev[l]
                    db = (self.razon_aprendizaje / N) * grad_b + self.momento * self.delta_sesgos_prev[l]

                    self.pesos[l] += dW
                    self.sesgos[l] += db

                    self.delta_pesos_prev[l] = dW
                    self.delta_sesgos_prev[l] = db

            # Medicion de error por epoca
            ecm_actual = self.calcular_ecm(X, d)
            self.historia_error.append(ecm_actual)

            if X_val is not None and d_val is not None:
                self.historia_val.append(self.calcular_ecm(X_val, d_val))

            if ecm_actual <= self.tolerancia:
                self.n_epocas_entrenadas = epoca
                self.motivo_parada = f"convergencia por tolerancia ({ecm_actual:.2e} <= {self.tolerancia:.2e})"
                return self

        self.n_epocas_entrenadas = self.max_epocas
        self.motivo_parada = f"limite maximo de epocas alcanzado ({self.max_epocas})"
        return self

    def resumen(self) -> str:
        """Cadena multilineal con el resumen estructural y estado de la red."""
        lineas = [
            f"Perceptron Multicapa {self.arquitectura}",
            f"  Capas: {len(self.arquitectura)} (entrada: {self.arquitectura[0]}, "
            f"ocultas: {self.arquitectura[1:-1]}, salida: {self.arquitectura[-1]})",
            f"  Activacion oculta: {self.act_oculta.nombre}",
            f"  Activacion salida: {self.act_salida.nombre}",
            f"  Razon de aprendizaje eta: {self.razon_aprendizaje}",
            f"  Momento alpha: {self.momento}",
            f"  Modo de entrenamiento: {self.modo}",
            f"  Epocas entrenadas: {self.n_epocas_entrenadas}",
            f"  Motivo de parada: {self.motivo_parada}",
            f"  ECM final: {self.historia_error[-1]:.6e}" if self.historia_error else "  No entrenado",
        ]
        return "\n".join(lineas)
