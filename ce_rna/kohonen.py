"""
Mapas Autoorganizados de Kohonen (Self-Organizing Maps, SOM).

Referencia teorica
------------------
* Kohonen, T. (1982): "Self-organized formation of topologically correct feature maps".
* `clases/claseRN02.md`, seccion "Aprendizaje no supervisado":
  "Las redes entrenadas sin supervision, tales como los mapas de Kohonen, son usados
  como vectores para formar cluster. Ninguna informacion relacionada a la clase correcta
  es dada a la red durante el entrenamiento."

Fundamentos matematicos
-----------------------
1. Topologia de la red:
   Una reticula bidimensional discreta de dimensiones (K_x, K_y) donde cada neurona j
   ocupa una coordenada fija r_j = (fila, columna) en el mapa y posee un vector de
   pesos sinapticos o prototipo w_j in R^D (con D la dimension del espacio de entrada).

2. Fase de Competicion (Best Matching Unit - BMU):
   Ante un vector de entrada x in R^D, se identifica la neurona ganadora j* cuya distancia
   euclidea al patron sea minima:
       j* = argmin_j || x - w_j ||

3. Fase de Cooperacion (Funcion de Vecindad):
   La activacion de la neurona ganadora influye sobre sus vecinas topologicas mediante
   una funcion gaussiana:
       h_{j, j*}(t) = exp( - || r_j - r_{j*} ||^2 / (2 * sigma(t)^2) )
   donde sigma(t) es el radio de vecindad en la epoca t, que decrece exponencialmente:
       sigma(t) = sigma_0 * (sigma_final / sigma_0)^(t / T)

4. Fase de Adaptacion:
   Los pesos de todas las neuronas se actualizan atrayendolos hacia la entrada x:
       w_j(t+1) = w_j(t) + eta(t) * h_{j, j*}(t) * (x - w_j(t))
   con la razon de aprendizaje eta(t) decreciendo exponencialmente:
       eta(t) = eta_0 * (eta_final / eta_0)^(t / T)

5. Metricas de Calidad del Mapa:
   - Error de Cuantizacion (QE): distancia media de cada patron a su BMU.
   - Error Topografico (TE): fraccion de patrones cuyo 1er y 2do BMU no son vecinos contiguos.
   - Matriz U (Unified Distance Matrix): distancia euclidea media entre cada neurona y sus
     vecinas inmediatas en la reticula, revelando fronteras y densidad de clusters.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class MapaAutoorganizado:
    """Mapa Autoorganizado de Kohonen (SOM) en reticula bidimensional rectangular.

    Parameters
    ----------
    forma_mapa:
        Tupla (filas, columnas) que define la reticula de neuronas. Ej. (6, 6).
    n_entradas:
        Dimension D del espacio de entrada (por ejemplo, 2 para puntos en el plano).
    razon_aprendizaje_inicial:
        eta_0, razon de aprendizaje inicial (0 < eta_0 <= 1).
    razon_aprendizaje_final:
        eta_final, razon de aprendizaje al terminar max_epocas.
    radio_inicial:
        sigma_0, radio inicial de vecindad. Si es None, se asigna max(forma_mapa) / 2.
    radio_final:
        sigma_final, radio de vecindad al terminar max_epocas.
    max_epocas:
        Numero total de epocas de entrenamiento.
    inicializacion:
        ``"aleatoria"`` (valores uniformes en el rango de los datos) o ``"pca"``.
    semilla:
        Semilla para el generador de numeros aleatorios (reproducibilidad).
    """

    def __init__(
        self,
        forma_mapa: tuple[int, int] = (6, 6),
        n_entradas: int = 2,
        razon_aprendizaje_inicial: float = 0.5,
        razon_aprendizaje_final: float = 0.01,
        radio_inicial: float | None = None,
        radio_final: float = 0.5,
        max_epocas: int = 200,
        inicializacion: str = "aleatoria",
        semilla: int | None = None,
    ):
        if len(forma_mapa) != 2 or any(k <= 0 for k in forma_mapa):
            raise ValueError("forma_mapa debe ser una tupla de dos enteros positivos (filas, columnas)")
        if n_entradas <= 0:
            raise ValueError("n_entradas debe ser un entero positivo")
        if not (0 < razon_aprendizaje_final <= razon_aprendizaje_inicial <= 1.0):
            raise ValueError("Las razones de aprendizaje deben cumplir 0 < final <= inicial <= 1")

        self.filas, self.columnas = forma_mapa
        self.n_neuronas = self.filas * self.columnas
        self.n_entradas = int(n_entradas)
        self.eta_0 = float(razon_aprendizaje_inicial)
        self.eta_f = float(razon_aprendizaje_final)
        self.sigma_0 = float(radio_inicial) if radio_inicial is not None else max(self.filas, self.columnas) / 2.0
        self.sigma_f = float(radio_final)
        self.max_epocas = int(max_epocas)
        self.inicializacion = inicializacion
        self.semilla = semilla

        self.rng = np.random.default_rng(semilla)

        # Coordenadas fijas en la reticula (filas, columnas) de cada neurona: shape (filas, columnas, 2)
        grid_r, grid_c = np.meshgrid(np.arange(self.filas), np.arange(self.columnas), indexing="ij")
        self.coordenadas_rejilla = np.stack([grid_r, grid_c], axis=-1)  # (filas, columnas, 2)

        # Matriz de pesos W: shape (filas, columnas, n_entradas)
        self.W = np.zeros((self.filas, self.columnas, self.n_entradas), dtype=float)

        self.historia_ec: list[float] = []  # Error de cuantizacion epoca a epoca
        self.entrenado = False

    def inicializar_pesos(self, X: np.ndarray) -> None:
        """Inicializa los pesos sinapticos cubriendo el rango de los datos."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        minimos = X.min(axis=0)
        maximos = X.max(axis=0)

        if self.inicializacion == "aleatoria":
            # Distribucion uniforme en la caja envolvente de X
            self.W = self.rng.uniform(minimos, maximos, size=(self.filas, self.columnas, self.n_entradas))
        elif self.inicializacion == "muestreo":
            # Muestreo aleatorio directo de puntos de datos
            indices = self.rng.choice(X.shape[0], size=(self.filas, self.columnas), replace=True)
            self.W = X[indices].copy()
        else:
            raise ValueError(f"Inicializacion desconocida: {self.inicializacion}")

    def bmu(self, x: np.ndarray) -> tuple[int, int]:
        """Encuentra la neurona ganadora (BMU) para un vector x de entrada."""
        # Distancia euclidea al cuadrado entre x y cada prototipo W[r, c]
        diff = self.W - x  # (filas, columnas, n_entradas)
        dist_sq = np.sum(diff ** 2, axis=-1)
        min_idx = np.argmin(dist_sq)
        r = min_idx // self.columnas
        c = min_idx % self.columnas
        return int(r), int(c)

    def bmu_top2(self, x: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """Encuentra las dos neuronas mas cercanas para el calculo del error topografico."""
        diff = self.W - x
        dist_sq = np.sum(diff ** 2, axis=-1).ravel()
        indices_ordenados = np.argsort(dist_sq)

        idx1, idx2 = indices_ordenados[0], indices_ordenados[1]
        bmu1 = (int(idx1 // self.columnas), int(idx1 % self.columnas))
        bmu2 = (int(idx2 // self.columnas), int(idx2 % self.columnas))
        return bmu1, bmu2

    def propagar(self, X: np.ndarray) -> np.ndarray:
        """Devuelve las coordenadas BMU (N, 2) de cada patron en X."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        bmus = np.empty((X.shape[0], 2), dtype=int)
        for i, x in enumerate(X):
            bmus[i] = self.bmu(x)
        return bmus

    def error_cuantizacion(self, X: np.ndarray) -> float:
        """Calcula el Error de Cuantizacion promedio (distancia de cada patron a su BMU)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        distancias = []
        for x in X:
            r, c = self.bmu(x)
            dist = np.linalg.norm(x - self.W[r, c])
            distancias.append(dist)
        return float(np.mean(distancias))

    def error_topografico(self, X: np.ndarray) -> float:
        """Calcula el Error Topografico (fraccion de patrones cuyos 2 mejores BMUs no son vecinos inmediatos)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        no_vecinos = 0
        for x in X:
            (r1, c1), (r2, c2) = self.bmu_top2(x)
            dist_topologica = abs(r1 - r2) + abs(c1 - c2)  # Distancia Manhattan en reticula
            if dist_topologica > 1:
                no_vecinos += 1
        return float(no_vecinos / X.shape[0])

    def entrenar(self, X: np.ndarray) -> MapaAutoorganizado:
        """Entrena el Mapa Autoorganizado de forma no supervisada sobre los datos X.

        Parameters
        ----------
        X:
            Matriz de datos (N, D).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_entradas:
            raise ValueError(f"X tiene {X.shape[1]} columnas; se esperaban {self.n_entradas}")

        self.inicializar_pesos(X)
        N = X.shape[0]
        T = self.max_epocas
        self.historia_ec = []

        # Registro inicial
        self.historia_ec.append(self.error_cuantizacion(X))

        for epoca in range(1, T + 1):
            # Parametros decrecientes exponenciales
            fraccion = epoca / T
            eta_t = self.eta_0 * ((self.eta_f / self.eta_0) ** fraccion)
            sigma_t = self.sigma_0 * ((self.sigma_f / self.sigma_0) ** fraccion)
            dos_sigma_sq = 2.0 * (sigma_t ** 2)

            # Barajar patrones en cada epoca
            indices = self.rng.permutation(N)

            for idx in indices:
                x = X[idx]

                # 1. Competicion: encontrar BMU
                r_bmu, c_bmu = self.bmu(x)

                # 2. Cooperacion: calcular vecindad gaussiana sobre la reticula
                dist_rejilla_sq = (
                    (self.coordenadas_rejilla[..., 0] - r_bmu) ** 2
                    + (self.coordenadas_rejilla[..., 1] - c_bmu) ** 2
                )
                h = np.exp(-dist_rejilla_sq / dos_sigma_sq)  # (filas, columnas)

                # 3. Adaptacion: actualizar pesos
                influencia = eta_t * h[..., np.newaxis]  # (filas, columnas, 1)
                self.W += influencia * (x - self.W)

            self.historia_ec.append(self.error_cuantizacion(X))

        self.entrenado = True
        return self

    def matriz_u(self) -> np.ndarray:
        """Calcula la Unified Distance Matrix (U-Matrix) para visualizar fronteras entre clusters.

        Para cada neurona (r, c), calcula la distancia euclidea media a sus vecinos contiguos
        (arriba, abajo, izquierda, derecha y diagonales si estan en el mapa).
        """
        U = np.zeros((self.filas, self.columnas), dtype=float)

        for r in range(self.filas):
            for c in range(self.columnas):
                vecinos = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.filas and 0 <= nc < self.columnas:
                            dist = np.linalg.norm(self.W[r, c] - self.W[nr, nc])
                            vecinos.append(dist)
                U[r, c] = np.mean(vecinos) if vecinos else 0.0

        return U

    def conteo_activaciones(self, X: np.ndarray) -> np.ndarray:
        """Devuelve una matriz (filas, columnas) con la frecuencia de aparicion de cada neurona como BMU."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        conteo = np.zeros((self.filas, self.columnas), dtype=int)
        for x in X:
            r, c = self.bmu(x)
            conteo[r, c] += 1
        return conteo

    def composicion_clusters(self, X: np.ndarray, etiquetas_reales: np.ndarray) -> dict[tuple[int, int], dict[int, int]]:
        """Mapea cada neurona (r, c) a los clusters reales que la activan (para analisis posterior)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(etiquetas_reales).ravel()
        composicion: dict[tuple[int, int], dict[int, int]] = {}

        for x, etiqueta in zip(X, y):
            coord = self.bmu(x)
            if coord not in composicion:
                composicion[coord] = {}
            composicion[coord][etiqueta] = composicion[coord].get(etiqueta, 0) + 1

        return composicion
