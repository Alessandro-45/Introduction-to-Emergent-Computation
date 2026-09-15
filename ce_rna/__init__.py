"""
ce_rna --- Redes Neuronales Artificiales desde cero
===================================================

Implementacion didactica, sin librerias de aprendizaje automatico, de los
modelos vistos en la asignatura *Introduccion a la Computacion Emergente*
(Prof. Esteban Alvarez).  Las unicas dependencias son `numpy` (algebra), `pandas`
(tablas) y `matplotlib` (figuras); todo el calculo neuronal --- propagacion,
reglas de aprendizaje, criterios de parada y metricas --- esta escrito a mano.

Modulos
-------
`activaciones`     Catalogo de funciones phi(v) y sus derivadas.
`datasets`         Conjuntos de entrenamiento de las clases (compuertas,
                   letras X/O, descodificador binario-decimal).
`mcculloch_pitts`  Neurona binaria de umbral fijo y redes compuestas (1943).
`perceptron`       Perceptron simple unicapa y regla perceptronica (1958).
`adaline`          Neurona lineal adaptativa y regla Delta (1960).
`hebb`             Regla de Hebb supervisada (1949).
`metricas`         Exactitud, matriz de confusion, ECM, RMSE, R2, margen.
`visual`           Diagramas de red, fronteras de decision, curvas y superficies.
`reportes`         Generacion automatica de los informes en Markdown.

Correspondencia con las clases
------------------------------
| Clase                 | Contenido                          | Modulo            |
|-----------------------|------------------------------------|-------------------|
| `claseRN01.md`        | Modelo de neurona, activaciones    | `activaciones`    |
| `claseRN02.md`        | McCulloch-Pitts, AND / OR          | `mcculloch_pitts` |
| `ICE-claseRN03.md`    | Separabilidad lineal, perceptron   | `perceptron`, `hebb` |
| `ICE-claseRN04.md`    | ADALINE y regla Delta              | `adaline`         |
"""

from . import (activaciones, adaline, datasets, hebb, mcculloch_pitts, metricas,
               perceptron, reportes, visual)

__version__ = "1.0.0"
__all__ = [
    "activaciones",
    "adaline",
    "datasets",
    "hebb",
    "mcculloch_pitts",
    "metricas",
    "perceptron",
    "reportes",
    "visual",
]
