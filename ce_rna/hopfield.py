"""
Red de Hopfield para Almacenamiento y Recuperacion de Patrones (Memoria Asociativa).

Referencia teorica
------------------
* Hopfield, J. J. (1982): "Neural networks and physical systems with emergent
  collective computational abilities".
* `clases/claseRN02.md`, seccion "Aprendizaje supervisado y memorias asociativas":
  "Las redes con supervision tales como la red de Hopfield y el perceptron son
  usadas como memorias asociativas o como clasificadores."
* Personnaz, Guyon y Dreyfus (1985): regla de proyeccion / pseudoinversa para
  patrones correlacionados.

Fundamentos matematicos
-----------------------
1. Arquitectura:
   Red recurrente de N neuronas completamente interconectadas con salidas bipolares
   s_i in {-1, +1}.
   - Conexiones simetricas: W_{ij} = W_{ji}.
   - Sin auto-conexiones: W_{ii} = 0.

2. Almacenamiento (Regla de Hebb):
   Para P patrones de memoria xi^mu in {-1, +1}^N, mu = 1, ..., P:
       W_{ij} = (1 / N) * sum_{mu=1}^P xi_i^mu * xi_j^mu  (para i != j)
       W_{ii} = 0
   En forma matricial:
       W = (1 / N) * sum_mu xi^mu (xi^mu)^T - (P / N) * I

3. Almacenamiento (Regla de Proyeccion / Pseudoinversa):
   Cuando los patrones presentan solapamiento correlacional (como letras del abecedario),
   la regla hebbiana sufre de interferencia cruzada (*cross-talk*). La regla del proyector
   ortogonal desacopla los patrones calculando:
       W = X^T * (X * X^T)^(-1) * X
   anulando la diagonal (W_{ii} = 0), lo que garantiza que todos los patrones almacenados
   sean puntos fijos estables (atractores exactos).

4. Dinamica de Recuperacion:
   Dado un estado inicial s(0) (patron incompleto o con ruido):
   - Actualizacion:
       s_i(t+1) = sgn( sum_{j=1}^N W_{ij} * s_j(t) - theta_i )
     con umbral theta_i = 0 y sgn(v) = +1 si v >= 0 y -1 si v < 0.
   - Modo Asincrono: actualiza neuronas una a una en orden aleatorio o secuencial.
     Garantiza convergencia a un minimo local de energia.
   - Modo Sincrono: actualiza todas las neuronas en paralelo s(t+1) = sgn(W * s(t)).

5. Funcion de Energia de Lyapunov:
   La red minimiza monotonamente la funcion de energia:
       E(s) = - 1/2 * sum_{i=1}^N sum_{j=1}^N W_{ij} * s_i * s_j = - 1/2 * s^T * W * s
   En cada paso de actualizacion asincrona, Delta E <= 0. Los patrones almacenados
   constituyen valles de atraccion (pozos de potencial) en el hiperespacio {-1, +1}^N.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class RedHopfield:
    """Red de Hopfield continua-discreta con funcion de energia de Lyapunov.

    Parameters
    ----------
    n_neuronas:
        Numero de neuronas N de la red (por ejemplo 42 para matrices de 7x6).
    regla:
        ``"hebb"`` utiliza la regla hebbiana estandar del modelo clasico.
        ``"proyeccion"`` utiliza la regla del proyector (pseudoinversa) para
        eliminar la interferencia cruzada entre patrones correlacionados.
    semilla:
        Semilla del generador aleatorio para la dinamica asincrona.
    """

    def __init__(
        self,
        n_neuronas: int = 42,
        regla: str = "proyeccion",
        semilla: int | None = None,
    ):
        if n_neuronas <= 1:
            raise ValueError("n_neuronas debe ser mayor a 1")
        if regla not in ("hebb", "proyeccion"):
            raise ValueError("regla debe ser 'hebb' o 'proyeccion'")

        self.n_neuronas = int(n_neuronas)
        self.regla = regla
        self.semilla = semilla
        self.rng = np.random.default_rng(semilla)

        # Matriz de pesos sinapticos W (N x N)
        self.W = np.zeros((self.n_neuronas, self.n_neuronas), dtype=float)
        self.patrones_almacenados: list[np.ndarray] = []
        self.nombres_patrones: list[str] = []

    def almacenar(
        self,
        patrones: Sequence[np.ndarray] | np.ndarray,
        nombres: Sequence[str] | None = None,
    ) -> RedHopfield:
        """Almacena un conjunto de patrones bipolares en la memoria asociativa.

        Parameters
        ----------
        patrones:
            Matriz de forma (P, N) o lista de P vectores de longitud N con valores {-1, +1}.
        nombres:
            Etiquetas opcionales para identificar los patrones almacenados.
        """
        X = np.atleast_2d(np.asarray(patrones, dtype=float))
        P, N = X.shape

        if N != self.n_neuronas:
            raise ValueError(f"Cada patron debe tener longitud {self.n_neuronas}; recibido {N}")

        # Validar bipolaridad
        valores_unicos = set(np.unique(X).tolist())
        if not valores_unicos.issubset({-1.0, 1.0}):
            raise ValueError("Los patrones de Hopfield deben ser estrictamente bipolares {-1.0, +1.0}")

        self.patrones_almacenados = [p.copy() for p in X]
        self.nombres_patrones = (
            list(nombres) if nombres is not None else [f"P{i+1}" for i in range(P)]
        )

        if self.regla == "hebb":
            # W = (1/N) * sum_mu xi_mu @ xi_mu^T
            self.W = np.zeros((N, N), dtype=float)
            for p in X:
                self.W += np.outer(p, p)
            self.W /= N
        else:  # Regla de proyeccion / pseudoinversa
            # W = X^T (X X^T)^(-1) X
            matriz_correlacion = X @ X.T
            inv_corr = np.linalg.pinv(matriz_correlacion)
            self.W = X.T @ inv_corr @ X

        # Condicion de Hopfield: conexiones simetricas sin auto-bucles (W_ii = 0)
        np.fill_diagonal(self.W, 0.0)
        self.W = 0.5 * (self.W + self.W.T)  # Garantizar simetria exacta

        return self

    def energia(self, s: np.ndarray) -> float:
        """Calcula la funcion de energia de Lyapunov: E(s) = -1/2 * s^T * W * s."""
        s = np.asarray(s, dtype=float).ravel()
        return float(-0.5 * (s @ self.W @ s))

    def recuperar(
        self,
        patron_inicial: np.ndarray,
        max_pasos: int = 50,
        modo: str = "asincrono",
    ) -> tuple[np.ndarray, list[float], bool]:
        """Evoluciona el estado desde patron_inicial hasta alcanzar un atractor estable.

        Parameters
        ----------
        patron_inicial:
            Vector bipolar (N,) de entrada (por ejemplo, imagen con ruido).
        max_pasos:
            Numero maximo de ciclos completos de actualizacion.
        modo:
            ``"asincrono"`` actualiza neuronas una a una en orden aleatorio (descenso estricto de energia).
            ``"sincrono"`` actualiza todas las neuronas a la vez.

        Returns
        -------
        estado_final:
            Vector bipolar recuperado.
        historia_energia:
            Lista con la evolucion de la energia de Lyapunov en cada paso.
        convergio:
            True si alcanzo un punto fijo estable.
        """
        s = np.asarray(patron_inicial, dtype=float).ravel().copy()
        N = self.n_neuronas
        historia_energia = [self.energia(s)]

        for paso in range(max_pasos):
            s_previo = s.copy()

            if modo == "asincrono":
                orden = self.rng.permutation(N)
                for i in orden:
                    campo_local = float(self.W[i] @ s)
                    s[i] = 1.0 if campo_local >= 0.0 else -1.0
            else:  # sincrono
                campo_total = self.W @ s
                s = np.where(campo_total >= 0.0, 1.0, -1.0)

            e_actual = self.energia(s)
            historia_energia.append(e_actual)

            if np.array_equal(s, s_previo):
                return s, historia_energia, True

        return s, historia_energia, False

    def identificar_patron(self, estado: np.ndarray) -> tuple[int | None, str, float]:
        """Compara el estado con los patrones almacenados mediante solapamiento normalizado.

        Returns
        -------
        indice:
            Indice del patron mas cercano, o None si no hay coincidencia positiva.
        nombre:
            Nombre del patron o "Estado Espurio".
        similitud:
            Fraccion de pixeles coincidentes (1.0 = recuperacion perfecta).
        """
        s = np.asarray(estado, dtype=float).ravel()
        mejor_indice = None
        mejor_nombre = "Estado Espurio"
        mejor_similitud = -1.0

        for idx, (p, nombre) in enumerate(zip(self.patrones_almacenados, self.nombres_patrones)):
            coincidencias = np.mean(s == p)
            if coincidencias > mejor_similitud:
                mejor_similitud = float(coincidencias)
                mejor_indice = idx
                mejor_nombre = nombre

        if mejor_similitud == 1.0:
            return mejor_indice, mejor_nombre, 1.0
        elif mejor_similitud > 0.8:
            return mejor_indice, f"{mejor_nombre} (parcial)", mejor_similitud
        else:
            return None, "Estado Espurio", mejor_similitud


# ---------------------------------------------------------------------------
# Utilidades de Imagen (PNG / Bipolar) y Ruido
# ---------------------------------------------------------------------------

def guardar_imagen_png(
    patron: np.ndarray,
    ruta: str | Path,
    forma: tuple[int, int] = (7, 6),
    escala: int = 24,
) -> Path:
    """Guarda un patron bipolar como un archivo de imagen PNG real visible.

    Parameters
    ----------
    patron:
        Vector bipolar (longitud filas x columnas) con valores {-1, +1}.
    ruta:
        Ruta del archivo .png de destino.
    forma:
        Dimensiones (filas, columnas) de la imagen (por omision 7x6).
    escala:
        Factor de escalado para que la imagen no sea diminuta (ej. 24x produce 144x168 px).
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(patron, dtype=float).reshape(forma)
    # Convencion visual de la clase: pixel encendido (+1) es oscuro (0), fondo (-1) es blanco (255)
    img_array = np.where(arr >= 0.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode="L")
    alto_px = forma[0] * escala
    ancho_px = forma[1] * escala
    img_escalada = img.resize((ancho_px, alto_px), resample=Image.Resampling.NEAREST)
    img_escalada.save(ruta)
    return ruta


def cargar_imagen_png(
    ruta: str | Path,
    forma: tuple[int, int] = (7, 6),
    umbral: int = 128,
) -> np.ndarray:
    """Lee un archivo de imagen real (PNG/JPG/BMP), lo convierte a escala de grises y lo binariza.

    Permite a la red de Hopfield recibir imagenes dibujadas o fotos externas.

    Parameters
    ----------
    ruta:
        Ruta del archivo de imagen.
    forma:
        Dimensiones (filas, columnas) a las que se redimensiona la imagen.
    umbral:
        Valor de brillo (0-255) para separar pixel encendido (+1) de fondo (-1).
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de imagen: {ruta}")

    img = Image.open(ruta).convert("L")
    # Redimensionar a (columnas, filas) segun la convencion de PIL (ancho, alto)
    img_res = img.resize((forma[1], forma[0]), resample=Image.Resampling.BILINEAR)
    arr = np.array(img_res)

    # Pixel oscuro (< umbral) es +1 (trazo), pixel claro es -1 (fondo)
    patron_bipolar = np.where(arr < umbral, 1.0, -1.0).ravel()
    return patron_bipolar


def contaminar_patron(patron: np.ndarray, n_pixeles: int, semilla: int | None = None) -> np.ndarray:
    """Invierte exactamente `n_pixeles` posiciones al azar en un patron bipolar."""
    rng = np.random.default_rng(semilla)
    copia = np.asarray(patron, dtype=float).copy().ravel()
    if n_pixeles > copia.size:
        raise ValueError(f"n_pixeles ({n_pixeles}) excede el tamano del patron ({copia.size})")
    indices = rng.choice(copia.size, size=n_pixeles, replace=False)
    copia[indices] *= -1.0
    return copia
