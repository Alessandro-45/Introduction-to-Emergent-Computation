"""
Metricas de evaluacion implementadas desde cero sobre `numpy`.

No se usa `scikit-learn` ni ninguna otra libreria de aprendizaje automatico:
todas las metricas se calculan a partir de operaciones vectorizadas basicas.
"""

from __future__ import annotations

import numpy as np


def exactitud(y_pred, y_real) -> float:
    """Fraccion de aciertos.  Acepta salidas vectoriales (todas deben coincidir)."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_real = np.asarray(y_real, dtype=float)
    if y_pred.ndim == 1:
        return float(np.mean(y_pred == y_real))
    return float(np.mean(np.all(y_pred == y_real, axis=1)))


def matriz_confusion(y_pred, y_real, etiquetas=None):
    """Matriz de confusion (filas = clase real, columnas = clase predicha)."""
    y_pred = np.asarray(y_pred).ravel()
    y_real = np.asarray(y_real).ravel()
    if etiquetas is None:
        etiquetas = np.unique(np.concatenate([y_real, y_pred]))
    etiquetas = np.asarray(etiquetas)
    indice = {v: k for k, v in enumerate(etiquetas)}
    M = np.zeros((etiquetas.size, etiquetas.size), dtype=int)
    for real, pred in zip(y_real, y_pred):
        M[indice[real], indice[pred]] += 1
    return M, etiquetas


def ecm(y_pred, y_real) -> float:
    """Error cuadratico medio con el factor 1/2 de la clase: (1/N) sum (1/2) e^2."""
    e = np.asarray(y_real, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(0.5 * e ** 2))


def rmse(y_pred, y_real) -> float:
    """Raiz del error cuadratico medio (sin el factor 1/2): unidades del objetivo."""
    e = np.asarray(y_real, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean(e ** 2)))


def error_absoluto_medio(y_pred, y_real) -> float:
    e = np.asarray(y_real, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(e)))


def r2(y_pred, y_real) -> float:
    """Coeficiente de determinacion R^2 = 1 - SSE/SST."""
    y_real = np.asarray(y_real, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    sse = float(np.sum((y_real - y_pred) ** 2))
    sst = float(np.sum((y_real - np.mean(y_real)) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def margen_geometrico(W, X, d) -> float:
    """Distancia minima de los patrones al hiperplano de separacion.

    Para un perceptron de una salida con pesos extendidos w = (w0, ..., wm), el
    margen del patron x es d * (w . x_ext) / ||w_{1:}||.  Un margen positivo en
    todos los patrones certifica que la separacion es correcta; su valor minimo
    mide cuan "holgada" es la solucion encontrada.
    """
    W = np.asarray(W, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    d = np.asarray(d, dtype=float).ravel()
    Xe = np.hstack([np.ones((X.shape[0], 1)), X])
    norma = np.linalg.norm(W[1:])
    if norma == 0:
        return 0.0
    return float(np.min(d * (Xe @ W)) / norma)
