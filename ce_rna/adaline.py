"""
ADALINE (ADAptive LInear NEuron, Widrow y Hoff 1960) y la regla Delta.

Referencia: `clases/ICE-claseRN04.md`.

Diferencia esencial con el perceptron
-------------------------------------
"El PERCEPTRON utiliza la salida de la funcion umbral para el aprendizaje; sin
embargo, la regla Delta utiliza directamente la salida de la red, sin pasarla
por ninguna funcion umbral."

Es decir, el ADALINE aprende sobre la salida **lineal**

    y^p = sum_{j=0}^{n} w_j x_j^p        (con x_0 = 1, el sesgo)

y minimiza el **error cuadratico medio** sobre el conjunto de patrones

    E = (1/N) sum_p E^p ,     E^p = (1/2) (d^p - y^p)^2

mediante **descenso del gradiente**.  Aplicando la regla de la cadena tal como
se desarrolla en la clase:

    dE^p/dw_j = (dE^p/dy^p) * (dy^p/dw_j) = -(d^p - y^p) * x_j

y por tanto el incremento de cada peso es

    Delta^p w_j = gamma * (d^p - y^p) * x_j

con gamma la razon de aprendizaje, "siempre un valor entre 0 y 1 para ponderar
el aprendizaje".

Nota sobre el enunciado del procedimiento
-----------------------------------------
El paso 5 del procedimiento de la clase dice "modificar el peso *restando* del
valor antiguo la cantidad obtenida en 4".  Eso es coherente solo si en el paso 4
se calcula la cantidad con el signo del gradiente, -(d-y)x; escrito con el
incremento Delta w_j = +gamma (d - y) x_j, el peso se **suma**.  Ambas
redacciones describen el mismo descenso del gradiente; aqui se implementa la
forma Delta w = +gamma (d - y) x, que es la que aparece en la formula final de
la propia clase.

Contenido del modulo
--------------------
* `Adaline`                    : la neurona lineal adaptativa y la regla Delta.
* `solucion_minimos_cuadrados` : la solucion analitica exacta (ecuaciones
  normales), que sirve de patron de oro para comprobar hasta donde converge el
  descenso del gradiente.
* `superficie_error`           : muestreo de E(w) en dos coordenadas de peso
  para dibujar el paraboloide de error y la trayectoria del aprendizaje.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HistorialEpocaAdaline:
    """Estado del ADALINE al terminar una epoca."""

    epoca: int
    w: np.ndarray
    ecm: float
    max_delta_peso: float


class Adaline:
    """Neurona lineal adaptativa entrenada con la regla Delta (Widrow-Hoff).

    Parameters
    ----------
    n_entradas:
        Numero de entradas sin contar el sesgo x_0 = 1.
    razon_aprendizaje:
        gamma, con 0 < gamma <= 1.  Es el parametro critico del modelo: si es
        demasiado grande el descenso del gradiente **diverge**.  Para el modo
        por lotes la cota teorica de estabilidad es gamma < 2 / lambda_max,
        donde lambda_max es el mayor autovalor de la matriz de correlacion
        (1/N) X^T X; el metodo `cota_estabilidad` la calcula.
    modo:
        ``"estocastico"`` actualiza los pesos patron a patron (LMS clasico, el
        procedimiento que describe la clase: "introducir un patron de entrada,
        calcular la salida ... modificar el peso ... regresar a 2");
        ``"lote"`` acumula el gradiente de todo el conjunto antes de actualizar
        (descenso del gradiente exacto sobre E).
    tolerancia:
        Criterio de convergencia: se detiene cuando el mayor cambio absoluto de
        un peso en una epoca cae por debajo de este valor.
    max_epocas:
        Numero maximo de barridos del conjunto de entrenamiento.
    inicializacion:
        ``"aleatoria"`` (la clase pide "inicializar los pesos de manera
        aleatoria de 0 a 1"), ``"ceros"`` o un vector explicito.
    barajar:
        Si se reordena el conjunto en cada epoca (solo en modo estocastico).
    semilla:
        Semilla del generador aleatorio.
    """

    def __init__(
        self,
        n_entradas: int,
        razon_aprendizaje: float = 0.01,
        modo: str = "estocastico",
        tolerancia: float = 1e-6,
        max_epocas: int = 200,
        inicializacion="aleatoria",
        barajar: bool = False,
        semilla: int = 0,
    ):
        if n_entradas < 1:
            raise ValueError("n_entradas debe ser >= 1")
        if razon_aprendizaje <= 0:
            raise ValueError("la razon de aprendizaje debe ser positiva")
        if modo not in ("estocastico", "lote"):
            raise ValueError("modo debe ser 'estocastico' o 'lote'")

        self.n_entradas = int(n_entradas)
        self.gamma = float(razon_aprendizaje)
        self.modo = modo
        self.tolerancia = float(tolerancia)
        self.max_epocas = int(max_epocas)
        self.barajar = bool(barajar)
        self.rng = np.random.default_rng(semilla)

        self.w = self._inicializar_pesos(inicializacion)
        self.w_inicial = self.w.copy()
        self.historial: list[HistorialEpocaAdaline] = []
        self.trayectoria_pesos: list[np.ndarray] = [self.w.copy()]
        #: Traza paso a paso: una fila por presentacion de patron.
        self.traza: list[dict] = []
        self.convergio = False
        self.diverged = False
        self.epocas_usadas = 0

    # -- inicializacion ----------------------------------------------------
    def _inicializar_pesos(self, inicializacion) -> np.ndarray:
        n = self.n_entradas + 1
        if isinstance(inicializacion, str):
            if inicializacion == "aleatoria":
                # "Inicializar los pesos de manera aleatoria de 0 a 1" (clase 04).
                return self.rng.uniform(0.0, 1.0, size=n)
            if inicializacion == "ceros":
                return np.zeros(n)
            raise ValueError("inicializacion debe ser 'aleatoria', 'ceros' o un vector")
        w = np.asarray(inicializacion, dtype=float)
        if w.shape != (n,):
            raise ValueError(f"el vector de pesos inicial debe tener forma ({n},)")
        return w.copy()

    # -- utilidades --------------------------------------------------------
    @staticmethod
    def _con_sesgo(X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([np.ones((X.shape[0], 1)), X])

    # -- inferencia --------------------------------------------------------
    def predecir(self, X) -> np.ndarray:
        """Salida lineal y = w . x  (sin funcion umbral)."""
        return self._con_sesgo(X) @ self.w

    __call__ = predecir

    def predecir_clase(self, X) -> np.ndarray:
        """Salida pasada por el escalon bipolar.

        El ADALINE *aprende* con la salida lineal, pero cuando se usa como
        clasificador la respuesta se obtiene aplicando el umbral a posteriori.
        Es la diferencia clave con el perceptron: el umbral interviene en la
        respuesta, nunca en el aprendizaje.
        """
        return np.where(self.predecir(X) >= 0.0, 1.0, -1.0)

    # -- medidas de error --------------------------------------------------
    def ecm(self, X, d) -> float:
        """Error cuadratico medio E = (1/N) sum_p (1/2)(d^p - y^p)^2."""
        d = np.asarray(d, dtype=float).ravel()
        e = d - self.predecir(X)
        return float(np.mean(0.5 * e ** 2))

    def error_por_patron(self, X, d) -> np.ndarray:
        d = np.asarray(d, dtype=float).ravel()
        return d - self.predecir(X)

    def cota_estabilidad(self, X) -> float:
        """Cota superior teorica de gamma para el modo por lotes: 2/lambda_max.

        lambda_max es el mayor autovalor de la matriz de correlacion de las
        entradas extendidas.  Por encima de esta cota el descenso del gradiente
        oscila con amplitud creciente y el ECM diverge.
        """
        Xe = self._con_sesgo(X)
        R = (Xe.T @ Xe) / Xe.shape[0]
        lambda_max = float(np.max(np.linalg.eigvalsh(R)))
        return 2.0 / lambda_max if lambda_max > 0 else np.inf

    # -- entrenamiento -----------------------------------------------------
    def entrenar(self, X, d, verbose: bool = False, registrar_pasos: bool = True,
                 registrar_traza: bool = False) -> "Adaline":
        """Aplica la regla Delta hasta converger o agotar `max_epocas`.

        Procedimiento de aprendizaje de `ICE-claseRN04.md`:

        1. Inicializar los pesos de manera aleatoria.
        2. Introducir un patron de entrada.
        3. Calcular la salida de la red, compararla con la deseada y obtener la
           diferencia (d^p - y^p).
        4. Para todos los pesos, multiplicar dicha diferencia por la entrada
           correspondiente y ponderarla por la tasa de aprendizaje.
        5. Modificar el peso con la cantidad obtenida en 4.
        6. Si no se ha cumplido el criterio de convergencia, regresar a 2.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = np.asarray(d, dtype=float).ravel()
        if X.shape[1] != self.n_entradas:
            raise ValueError(
                f"la red espera {self.n_entradas} entradas, recibio {X.shape[1]}"
            )
        if X.shape[0] != d.size:
            raise ValueError("X y d deben tener el mismo numero de patrones")

        Xe = self._con_sesgo(X)
        N = Xe.shape[0]

        self.historial = []
        self.trayectoria_pesos = [self.w.copy()]
        self.traza = []
        self.convergio = False
        self.diverged = False

        for epoca in range(1, self.max_epocas + 1):
            w_previo = self.w.copy()
            orden = self.rng.permutation(N) if (self.barajar and self.modo == "estocastico") else np.arange(N)

            if self.modo == "estocastico":
                for n in orden:
                    x = Xe[n]
                    y = float(x @ self.w)          # paso 3: salida lineal
                    error = d[n] - y               # paso 3: diferencia (d - y)
                    self.w = self.w + self.gamma * error * x   # pasos 4 y 5
                    if registrar_pasos:
                        self.trayectoria_pesos.append(self.w.copy())
                    if registrar_traza:
                        fila = {"epoca": epoca, "patron": int(n) + 1}
                        for i in range(1, x.size):
                            fila[f"x{i}"] = float(x[i])
                        fila["d"] = float(d[n])
                        fila["y"] = y
                        fila["error"] = error
                        fila["error^2/2"] = 0.5 * error ** 2
                        for i in range(x.size):
                            fila[f"w{i}"] = float(self.w[i])
                        self.traza.append(fila)
            else:  # lote: gradiente exacto de E
                y = Xe @ self.w
                error = d - y
                gradiente = -(Xe.T @ error) / N
                self.w = self.w - self.gamma * gradiente
                if registrar_pasos:
                    self.trayectoria_pesos.append(self.w.copy())

            ecm = self.ecm(X, d)
            max_delta = float(np.max(np.abs(self.w - w_previo)))
            self.historial.append(
                HistorialEpocaAdaline(epoca=epoca, w=self.w.copy(), ecm=ecm, max_delta_peso=max_delta)
            )
            if verbose:
                pesos = " ".join(f"{v:7.4f}" for v in self.w)
                print(f"  epoca {epoca:4d} | ECM={ecm:12.6e} | w=[{pesos}]")

            self.epocas_usadas = epoca

            if not np.isfinite(ecm) or ecm > 1e12:
                self.diverged = True
                break
            if max_delta < self.tolerancia:
                self.convergio = True
                break

        return self

    # -- diagnostico -------------------------------------------------------
    def resumen(self) -> str:
        if self.diverged:
            estado = "DIVERGIO (razon de aprendizaje demasiado grande)"
        elif self.convergio:
            estado = "CONVERGIO"
        else:
            estado = "NO convergio (limite de epocas)"
        pesos = "  ".join(f"{v:9.5f}" for v in self.w)
        return (
            f"ADALINE  n={self.n_entradas}  gamma={self.gamma}  modo={self.modo}\n"
            f"Estado: {estado} en {self.epocas_usadas} epocas\n"
            f"Pesos finales (w0 = sesgo): {pesos}"
        )

    def traza_df(self):
        """Traza paso a paso como `pandas.DataFrame` (requiere `registrar_traza=True`).

        Responde literalmente a la consigna de `ICE-claseRN04.md`: *"refleje en
        pantalla de manera secuencial el numero de iteracion, pesos y el error
        por patron"*.
        """
        import pandas as pd

        return pd.DataFrame(self.traza)

    def historial_df(self):
        import pandas as pd

        datos = {
            "epoca": [h.epoca for h in self.historial],
            "ecm": [h.ecm for h in self.historial],
            "max_delta_peso": [h.max_delta_peso for h in self.historial],
        }
        W = np.array([h.w for h in self.historial])
        for i in range(W.shape[1]):
            datos[f"w{i}"] = W[:, i]
        return pd.DataFrame(datos)


# ---------------------------------------------------------------------------
# Solucion analitica de referencia
# ---------------------------------------------------------------------------

def solucion_minimos_cuadrados(X, d) -> np.ndarray:
    """Minimo global de E(w) resolviendo las ecuaciones normales X^T X w = X^T d.

    La superficie de error de un ADALINE es un paraboloide convexo, de modo que
    tiene un unico minimo (o un subespacio de minimos si X^T X es singular).
    El descenso del gradiente de la regla Delta converge a este punto; tenerlo
    calculado de forma exacta permite medir cuanto le falta al aprendizaje.

    Se usa `numpy.linalg.lstsq`, que maneja el caso singular devolviendo la
    solucion de norma minima.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = np.asarray(d, dtype=float).ravel()
    Xe = np.hstack([np.ones((X.shape[0], 1)), X])
    w, *_ = np.linalg.lstsq(Xe, d, rcond=None)
    return w


def ecm_de_pesos(w, X, d) -> float:
    """ECM de un vector de pesos arbitrario (sin necesidad de instanciar el modelo)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = np.asarray(d, dtype=float).ravel()
    Xe = np.hstack([np.ones((X.shape[0], 1)), X])
    e = d - Xe @ np.asarray(w, dtype=float)
    return float(np.mean(0.5 * e ** 2))


def superficie_error(X, d, indices=(1, 2), w_base=None, rango=4.0, n_puntos=80):
    """Muestrea E(w) variando dos componentes de w y fijando las demas.

    Devuelve (W_i, W_j, E), tres matrices listas para `contour` o `plot_surface`.
    Sirve para visualizar el paraboloide de error sobre el que "desciende" la
    regla Delta.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = np.asarray(d, dtype=float).ravel()
    i, j = indices
    if w_base is None:
        w_base = solucion_minimos_cuadrados(X, d)
    w_base = np.asarray(w_base, dtype=float).copy()

    eje_i = np.linspace(w_base[i] - rango, w_base[i] + rango, n_puntos)
    eje_j = np.linspace(w_base[j] - rango, w_base[j] + rango, n_puntos)
    Wi, Wj = np.meshgrid(eje_i, eje_j)

    E = np.empty_like(Wi)
    w = w_base.copy()
    for fila in range(Wi.shape[0]):
        for col in range(Wi.shape[1]):
            w[i] = Wi[fila, col]
            w[j] = Wj[fila, col]
            E[fila, col] = ecm_de_pesos(w, X, d)
    return Wi, Wj, E
