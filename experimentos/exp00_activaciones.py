"""
Experimento 00 --- Catalogo de funciones de activacion.

Genera el panel de figuras de las funciones de activacion enumeradas en
`claseRN01.md` y verifica numericamente sus propiedades: recorrido, simetria y
coincidencia de la derivada analitica con la diferencia finita.

Ejecucion
---------
    python experimentos/exp00_activaciones.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import activaciones as act
from ce_rna import visual as vz
from ce_rna.reportes import Reporte, encabezado_experimento


def main() -> None:
    encabezado_experimento("Experimento 00 - Funciones de activacion")

    reporte = Reporte(
        titulo="Experimento 00 --- Catalogo de funciones de activacion",
        nombre_archivo="00_activaciones",
        resumen=(
            "Las funciones de activacion determinan *\"el nivel de activacion de la neurona en "
            "terminos de la actividad existente en sus entradas\"*. Este experimento dibuja el "
            "catalogo completo de `claseRN01.md` y comprueba una propiedad que condiciona todo "
            "el resto del repositorio: cuales son derivables y cuales no."
        ),
    )

    fig, _ = vz.dibujar_activaciones()
    ruta = vz.guardar(fig, FIGURAS / "00_activaciones.png")
    reporte.figura(ruta, "Las seis funciones de activacion del catalogo del curso.")

    v = np.linspace(-4.0, 4.0, 2001)
    h = 1e-6
    filas = []
    for phi in (act.Escalon(), act.EscalonBipolar(), act.EscalonBipolarConZona(1.0),
                act.Identidad(), act.Sigmoide(), act.SigmoideBipolar(), act.Gaussiana(0.0, 1.5)):
        salida = phi(v)
        if phi.derivable:
            numerica = (phi(v + h) - phi(v - h)) / (2 * h)
            error = float(np.max(np.abs(phi.derivada(v) - numerica)))
            comprobacion = f"{error:.2e}"
        else:
            comprobacion = "no derivable"
        filas.append({
            "activacion": phi.nombre,
            "minimo": float(np.min(salida)),
            "maximo": float(np.max(salida)),
            "valores distintos": int(np.unique(np.round(salida, 9)).size),
            "derivable": "si" if phi.derivable else "no",
            "|dphi analitica - numerica|": comprobacion,
        })
    df = pd.DataFrame(filas)
    print(df.to_string(index=False))
    reporte.tabla(df, "Propiedades verificadas numericamente", nombre_csv="00_activaciones")

    reporte.seccion("Consecuencia para el aprendizaje")
    reporte.texto(
        "Las tres variantes del escalon tienen derivada nula en todo punto donde esta definida. "
        "Eso significa que **el descenso del gradiente no puede aplicarse** a una neurona con "
        "activacion escalon: el gradiente del error respecto a los pesos es identicamente cero y "
        "no senala ninguna direccion de mejora. De ahi las dos familias de reglas de aprendizaje "
        "del repositorio: las **discretas de correccion de error** (perceptron, Hebb), que no "
        "derivan de ningun gradiente, y las **de gradiente** (regla Delta del ADALINE), que exigen "
        "una salida derivable y por eso aprenden sobre la salida lineal, aplicando el umbral solo "
        "despues del aprendizaje."
    )

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
