"""
Perceptron simple (red unicapa) con la regla de aprendizaje perceptronica.

Referencia: `clases/ICE-claseRN03.md`, secciones "El perceptron" y "Algoritmo
Perceptronico"; y `clases/claseRN02.md`, seccion "El perceptron".

Arquitectura
------------
Una unica capa de m1 neuronas de salida no lineales, alimentada por m0 entradas
mas la "neurona de inclinacion" x0 = 1 que sustituye al umbral::

    y_j^(in)(n) = sum_{i=0}^{m0} w_ji(n) * x_i(n)          (potencial)
    y_j(n)      = phi( y_j^(in)(n) )                       (salida bipolar)

Las sinapsis forman una matriz `W` de n x (m+1) tal como indica la clase: "las
sinapsis obviamente estan ordenadas en una matriz wji de n*(m+1)".  Aqui
`W[j, 0]` es el peso de inclinacion (sesgo) de la neurona j.

Algoritmo perceptronico (pasos 0 a 5 de la clase)
-------------------------------------------------
* **Paso 0**: inicializar las sinapsis (a cero o con valores aleatorios) y
  elegir una razon de aprendizaje 0 < a < 1.
* **Paso 1**: mientras la condicion de parada del paso 5 sea falsa, repetir.
* **Paso 2**: para cada par de entrenamiento (x_i(n), d_j(n)).
* **Paso 3**: calcular y_j^(in)(n) y y_j(n).
* **Paso 4**: si y_j != d_j entonces w_ji(n+1) = w_ji(n) + a * d_j * x_i(n);
  en caso contrario los pesos no cambian.
* **Paso 5**: si los pesos no cambiaron durante un barrido completo del conjunto
  de entrenamiento, parar.

Observaciones importantes que se verifican en los experimentos
--------------------------------------------------------------
1. El incremento usa **d_j** (la salida deseada) y no el error d_j - y_j; esta
   es la forma clasica de Rosenblatt que emplea la clase.  Como la correccion
   solo se aplica cuando hay error, y en codificacion bipolar el error vale
   d_j - y_j = 2*d_j cuando y_j = -d_j, ambas formulaciones difieren unicamente
   en un factor constante absorbible por la razon de aprendizaje... salvo
   cuando la salida cae en el punto de indeterminacion (y_j = 0), donde el
   error vale d_j y ambas coinciden exactamente.
2. Si el problema es linealmente separable, el algoritmo **converge en un
   numero finito de pasos** (teorema de convergencia del perceptron).  Si no lo
   es --- el XOR --- el algoritmo no se detiene nunca y por eso hace falta un
   limite de epocas.
3. La solucion no es unica: "o no existe ninguna solucion, o existen
   infinitas" (`ICE-claseRN03.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activaciones import EscalonBipolar, EscalonBipolarConZona, FuncionActivacion


@dataclass
class HistorialEpoca:
    """Instantanea del estado de la red al terminar una epoca."""

    epoca: int
    W: np.ndarray
    n_actualizaciones: int
    n_patrones_mal: int
    exactitud: float


class PerceptronSimple:
    """Red unicapa de m1 perceptrones entrenada con la regla perceptronica.

    Parameters
    ----------
    n_entradas:
        m0, numero de entradas sin contar la inclinacion.
    n_salidas:
        m1, numero de neuronas de salida.  Cada neurona de salida representa una
        clase: "si una de ellas se activa con una entrada, significa que
        pertenece a esa clase, si esta desactivada, que no pertenece".
    razon_aprendizaje:
        a, con 0 < a <= 1 (paso 0 del algoritmo).
    theta:
        Semiancho del punto de indeterminacion de la activacion bipolar.  Con
        theta = 0 la activacion es el escalon bipolar puro.
    inicializacion:
        ``"ceros"`` (opcion explicita de la clase: "se puede elegir wji = 0") o
        ``"aleatoria"`` (valores uniformes pequenos).
    max_epocas:
        Cota de seguridad para problemas no separables, donde el criterio de
        parada del paso 5 nunca se cumple.
    actualizar_todas_las_salidas:
        Lectura literal del paso 4 de la clase ("si yj != dj para *algun* j,
        entonces ... donde j = 1,...,m1"): cuando una sola neurona se equivoca
        se corrigen todas.  Por omision es ``False``, que es la formulacion
        habitual y la unica que garantiza el teorema de convergencia: cada
        neurona de salida es un perceptron independiente y solo se corrige la
        que falla.
    semilla:
        Semilla del generador aleatorio (reproducibilidad).
    """

    def __init__(
        self,
        n_entradas: int,
        n_salidas: int = 1,
        razon_aprendizaje: float = 1.0,
        theta: float = 0.0,
        inicializacion: str = "ceros",
        max_epocas: int = 100,
        actualizar_todas_las_salidas: bool = False,
        semilla: int = 0,
    ):
        if n_entradas < 1 or n_salidas < 1:
            raise ValueError("n_entradas y n_salidas deben ser >= 1")
        if not 0 < razon_aprendizaje <= 1:
            raise ValueError("la razon de aprendizaje debe cumplir 0 < a <= 1")

        self.n_entradas = int(n_entradas)
        self.n_salidas = int(n_salidas)
        self.a = float(razon_aprendizaje)
        self.theta = float(theta)
        self.max_epocas = int(max_epocas)
        self.actualizar_todas_las_salidas = bool(actualizar_todas_las_salidas)
        self.inicializacion = inicializacion
        self.rng = np.random.default_rng(semilla)

        self.phi: FuncionActivacion = (
            EscalonBipolarConZona(theta) if theta > 0 else EscalonBipolar()
        )

        self.W = self._inicializar_pesos()
        self.historial: list[HistorialEpoca] = []
        self.trayectoria_pesos: list[np.ndarray] = [self.W.copy()]
        #: Traza paso a paso del algoritmo (una fila por presentacion de patron).
        self.traza: list[dict] = []
        self.convergio = False
        self.epocas_usadas = 0

    # -- inicializacion ----------------------------------------------------
    def _inicializar_pesos(self) -> np.ndarray:
        """Paso 0: "inicializar las sinapsis de la red, se puede elegir wji=0
        o valores aleatorios"."""
        forma = (self.n_salidas, self.n_entradas + 1)
        if self.inicializacion == "ceros":
            return np.zeros(forma)
        if self.inicializacion == "aleatoria":
            return self.rng.uniform(-0.5, 0.5, size=forma)
        raise ValueError("inicializacion debe ser 'ceros' o 'aleatoria'")

    # -- utilidades internas ----------------------------------------------
    @staticmethod
    def _con_inclinacion(X: np.ndarray) -> np.ndarray:
        """Antepone la columna x0 = 1 (la neurona de inclinacion)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _objetivo_2d(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        if d.shape[1] != self.n_salidas:
            raise ValueError(
                f"la red tiene {self.n_salidas} salidas, los objetivos tienen {d.shape[1]}"
            )
        return d

    # -- inferencia --------------------------------------------------------
    def potencial(self, X) -> np.ndarray:
        """y^(in) = X_ext @ W.T   ->  matriz (N, m1)."""
        return self._con_inclinacion(X) @ self.W.T

    def predecir(self, X) -> np.ndarray:
        """Salida bipolar de la red.  Devuelve (N,) si hay una sola salida."""
        y = self.phi(self.potencial(X))
        return y[:, 0] if self.n_salidas == 1 else y

    __call__ = predecir

    def clasificar(self, X) -> np.ndarray:
        """Indice de la neurona de salida con mayor potencial (competicion).

        Util cuando varias neuronas se activan a la vez o ninguna lo hace: en
        lugar de leer el patron bipolar de salida, se toma la clase cuyo
        potencial y^(in) es maximo.
        """
        return np.argmax(self.potencial(X), axis=1)

    # -- entrenamiento -----------------------------------------------------
    def entrenar(self, X, d, verbose: bool = False, registrar_traza: bool = False) -> "PerceptronSimple":
        """Ejecuta el algoritmo perceptronico (pasos 1 a 5).

        Los patrones se presentan **uno a uno y en orden** (aprendizaje
        estocastico o "en linea"), tal como describe el paso 2 de la clase.

        Con `registrar_traza=True` se guarda en `self.traza` una fila por cada
        presentacion de patron con el potencial, la salida, la salida deseada y
        los pesos antes y despues de la correccion.  Es el registro que permite
        auditar el algoritmo paso a paso en los informes.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        d = self._objetivo_2d(d)
        if X.shape[1] != self.n_entradas:
            raise ValueError(
                f"la red espera {self.n_entradas} entradas, recibio {X.shape[1]}"
            )
        Xe = self._con_inclinacion(X)

        self.historial = []
        self.trayectoria_pesos = [self.W.copy()]
        self.traza = []
        self.convergio = False

        for epoca in range(1, self.max_epocas + 1):
            n_actualizaciones = 0

            # Paso 2: para cada par de entrenamiento (x_i(n), d_j(n)).
            for n in range(Xe.shape[0]):
                x = Xe[n]
                # Paso 3: barrido sobre las neuronas de salida.
                potencial = self.W @ x
                y = self.phi(potencial)
                objetivo = d[n]
                fallan = y != objetivo
                W_previo = self.W.copy() if registrar_traza else None

                # Paso 4: correccion solo si hay error.
                if np.any(fallan):
                    indices = (
                        np.arange(self.n_salidas)
                        if self.actualizar_todas_las_salidas
                        else np.flatnonzero(fallan)
                    )
                    self.W[indices] += self.a * np.outer(objetivo[indices], x)
                    n_actualizaciones += 1
                    self.trayectoria_pesos.append(self.W.copy())

                if registrar_traza:
                    fila = {"epoca": epoca, "patron": n + 1}
                    for i in range(1, x.size):
                        fila[f"x{i}"] = float(x[i])
                    for j in range(self.n_salidas):
                        sufijo = "" if self.n_salidas == 1 else f"_{j}"
                        fila[f"y_in{sufijo}"] = float(potencial[j])
                        fila[f"y{sufijo}"] = float(y[j])
                        fila[f"d{sufijo}"] = float(objetivo[j])
                    fila["corrige"] = "si" if np.any(fallan) else "no"
                    for j in range(self.n_salidas):
                        for i in range(x.size):
                            sufijo = "" if self.n_salidas == 1 else f"_{j}"
                            fila[f"w{i}{sufijo}"] = float(self.W[j, i])
                    self.traza.append(fila)

            # Diagnostico de la epoca.
            y_total = self.phi(Xe @ self.W.T)
            patrones_mal = int(np.sum(np.any(y_total != d, axis=1)))
            exactitud = 1.0 - patrones_mal / Xe.shape[0]
            self.historial.append(
                HistorialEpoca(
                    epoca=epoca,
                    W=self.W.copy(),
                    n_actualizaciones=n_actualizaciones,
                    n_patrones_mal=patrones_mal,
                    exactitud=exactitud,
                )
            )
            if verbose:
                print(
                    f"  epoca {epoca:3d} | actualizaciones={n_actualizaciones:3d} "
                    f"| patrones mal={patrones_mal:3d} | exactitud={exactitud:6.2%}"
                )

            self.epocas_usadas = epoca
            # Paso 5: si los pesos no cambiaron en toda la epoca, parar.
            if n_actualizaciones == 0:
                self.convergio = True
                break

        return self

    # -- diagnostico -------------------------------------------------------
    def exactitud(self, X, d) -> float:
        """Fraccion de patrones cuya salida completa coincide con la deseada."""
        d = self._objetivo_2d(d)
        y = self.phi(self.potencial(X))
        return float(np.mean(np.all(y == d, axis=1)))

    def errores_por_patron(self, X, d) -> np.ndarray:
        d = self._objetivo_2d(d)
        y = self.phi(self.potencial(X))
        return np.any(y != d, axis=1)

    def recta_frontera(self, neurona: int = 0) -> tuple[float, float, float]:
        """Coeficientes (w0, w1, w2) de la frontera w0 + w1 x1 + w2 x2 = 0.

        Solo tiene sentido para redes de dos entradas.  `ICE-claseRN03.md`:
        "la frontera entre ambas esta dada por la ecuacion lineal de la recta
        w0 + w1 x1 + w2 x2 = 0".
        """
        if self.n_entradas != 2:
            raise ValueError("la recta frontera solo esta definida para 2 entradas")
        w = self.W[neurona]
        return float(w[0]), float(w[1]), float(w[2])

    def resumen(self) -> str:
        estado = "CONVERGIO" if self.convergio else "NO convergio (limite de epocas)"
        lineas = [
            f"Perceptron simple  m0={self.n_entradas}  m1={self.n_salidas}  a={self.a}  theta={self.theta}",
            f"Estado: {estado} en {self.epocas_usadas} epocas "
            f"({len(self.trayectoria_pesos) - 1} actualizaciones de pesos)",
            "Pesos finales (fila = neurona, columna 0 = inclinacion):",
        ]
        for j in range(self.n_salidas):
            pesos = "  ".join(f"{w:8.3f}" for w in self.W[j])
            lineas.append(f"  neurona {j}: {pesos}")
        return "\n".join(lineas)

    def traza_df(self):
        """Traza paso a paso como `pandas.DataFrame` (requiere `registrar_traza=True`)."""
        import pandas as pd

        return pd.DataFrame(self.traza)

    def historial_df(self):
        """Historial de entrenamiento como `pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(
            {
                "epoca": [h.epoca for h in self.historial],
                "actualizaciones": [h.n_actualizaciones for h in self.historial],
                "patrones_mal": [h.n_patrones_mal for h in self.historial],
                "exactitud": [h.exactitud for h in self.historial],
            }
        )
