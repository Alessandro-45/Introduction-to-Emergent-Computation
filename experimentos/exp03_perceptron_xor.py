"""
Experimento 03 --- El limite del perceptron simple: el XOR.

`ICE-claseRN03.md` afirma que el XOR no es linealmente separable y que por eso
"no nos basta con una red sencilla para resolverlo".  Este experimento convierte
esa afirmacion en evidencia numerica:

1. Se entrena el perceptron sobre el XOR y se comprueba que el paso 5 (criterio
   de parada) **nunca** se cumple: el algoritmo oscila indefinidamente.
2. Se registra la exactitud epoca a epoca para ver que se estanca en 50-75 %.
3. Se explora exhaustivamente el espacio de pesos (w0, w1, w2) sobre una rejilla
   fina y se verifica que **ninguna** recta clasifica los cuatro patrones.
4. Se contrasta con AND y OR, que si convergen, y se senala la salida: anadir
   capas (red multicapa) o componer neuronas a mano (experimento 01).

Ejecucion
---------
    python experimentos/exp03_perceptron_xor.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.mcculloch_pitts import red_xor
from ce_rna.perceptron import PerceptronSimple
from ce_rna.reportes import Reporte, encabezado_experimento


def busqueda_exhaustiva_recta(conjunto, n=41, limite=3.0) -> tuple[float, np.ndarray]:
    """Recorre una rejilla de rectas w0 + w1 x1 + w2 x2 = 0 y devuelve la mejor.

    Devuelve (mejor_exactitud, mejores_pesos).  Con 41^3 = 68 921 combinaciones
    la rejilla es suficientemente fina como para que el resultado sea
    concluyente en la practica; la imposibilidad formal proviene de la no
    separabilidad lineal, no del muestreo.
    """
    ejes = np.linspace(-limite, limite, n)
    mejor = (-1.0, None)
    Xe = np.hstack([np.ones((conjunto.X.shape[0], 1)), conjunto.X])
    for w0 in ejes:
        for w1 in ejes:
            for w2 in ejes:
                w = np.array([w0, w1, w2])
                y = np.where(Xe @ w >= 0, 1.0, -1.0)
                exactitud = float(np.mean(y == conjunto.d))
                if exactitud > mejor[0]:
                    mejor = (exactitud, w)
    return mejor


def main() -> None:
    encabezado_experimento("Experimento 03 - El limite del perceptron simple: XOR")

    reporte = Reporte(
        titulo="Experimento 03 --- El limite del perceptron simple: el problema XOR",
        nombre_archivo="03_perceptron_xor",
        resumen=(
            "Un perceptron de una capa solo puede trazar un hiperplano, de modo que solo "
            "resuelve problemas **linealmente separables**. El XOR no lo es. Este experimento "
            "documenta que el algoritmo perceptronico no converge sobre el XOR, mide hasta "
            "donde llega, y demuestra por busqueda exhaustiva sobre el espacio de pesos que el "
            "fallo no es del algoritmo sino de la arquitectura."
        ),
    )

    conjunto = ds.compuerta("XOR", "bipolar")

    # ------------------------------------------------------------------
    # 1. No separabilidad
    # ------------------------------------------------------------------
    reporte.seccion("1. El XOR sobre el plano")
    reporte.tabla(conjunto.tabla(), "Tabla de verdad del XOR en codificacion bipolar",
                  nombre_csv="03_tabla_xor")
    reporte.texto(
        "Los patrones de la clase +1 --- (−1,+1) y (+1,−1) --- ocupan una diagonal del cuadrado, "
        "y los de la clase −1 la otra. Cualquier recta que deje las dos esquinas de una diagonal "
        "a un lado deja tambien, necesariamente, al menos una de la otra diagonal."
    )

    # ------------------------------------------------------------------
    # 2. El algoritmo no converge
    # ------------------------------------------------------------------
    reporte.seccion("2. El algoritmo perceptronico no se detiene")
    modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=200)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_traza=True)
    historial = modelo.historial_df()

    print(f"\nEstado tras {modelo.max_epocas} epocas: convergio = {modelo.convergio}")
    print(f"Exactitud final: {mt.exactitud(modelo.predecir(conjunto.X), conjunto.d):.0%}")
    print(f"Exactitud maxima alcanzada en alguna epoca: {historial['exactitud'].max():.0%}")
    print(f"Correcciones totales: {len(modelo.trayectoria_pesos) - 1}")

    reporte.texto(
        f"Con pesos iniciales nulos y a = 1 el algoritmo realiza "
        f"**{len(modelo.trayectoria_pesos) - 1} correcciones en {modelo.epocas_usadas} epocas** y "
        f"**nunca** satisface el paso 5 (`convergio = {modelo.convergio}`). La ejecucion termina "
        "unicamente porque se alcanza el limite de epocas, que es una salvaguarda del programa, "
        "no un criterio del algoritmo."
    )
    reporte.tabla(historial.head(12), "Primeras 12 epocas: la exactitud oscila y no mejora",
                  nombre_csv="03_historial_xor")

    # Varias corridas: el resultado no depende de la inicializacion ni de a
    filas_corridas = []
    for a in (0.2, 0.5, 1.0):
        for inic, semilla in (("ceros", 0), ("aleatoria", 1), ("aleatoria", 2)):
            m = PerceptronSimple(2, 1, razon_aprendizaje=a, inicializacion=inic,
                                 semilla=semilla, max_epocas=200)
            m.entrenar(conjunto.X, conjunto.d)
            h = m.historial_df()
            filas_corridas.append({
                "a": a,
                "inicializacion": f"{inic}" + ("" if inic == "ceros" else f" (semilla {semilla})"),
                "convergio": "si" if m.convergio else "no",
                "epocas": m.epocas_usadas,
                "correcciones": len(m.trayectoria_pesos) - 1,
                "exactitud_final": mt.exactitud(m.predecir(conjunto.X), conjunto.d),
                "exactitud_maxima": float(h["exactitud"].max()),
            })
    df_corridas = pd.DataFrame(filas_corridas)
    print("\nNueve corridas con distintas razones de aprendizaje e inicializaciones:")
    print(df_corridas.to_string(index=False))
    reporte.tabla(df_corridas, "Ninguna combinacion de parametros converge",
                  nombre_csv="03_corridas_xor")
    reporte.texto(
        f"Nueve corridas con tres razones de aprendizaje y tres inicializaciones distintas: "
        f"**ninguna converge**. La exactitud maxima observada en alguna epoca es "
        f"{df_corridas['exactitud_maxima'].max():.0%} y la final oscila entre "
        f"{df_corridas['exactitud_final'].min():.0%} y {df_corridas['exactitud_final'].max():.0%}. "
        "El resultado no depende de como se ajuste el algoritmo."
    )

    fig, _ = vz.curva_aprendizaje(
        historial["patrones_mal"].to_numpy(),
        titulo="XOR: patrones mal clasificados por epoca (no converge)",
        etiqueta_y="patrones mal clasificados (de 4)",
        color=vz.NARANJA,
    )
    ruta = vz.guardar(fig, FIGURAS / "03_xor_error_por_epoca.png")
    reporte.figura(ruta, "El error nunca llega a cero: el ciclo de correcciones se repite indefinidamente.")

    fig, _ = vz.frontera_decision(
        modelo.W[0], conjunto.X, conjunto.d,
        titulo="XOR: la mejor recta que el perceptron llega a proponer",
        etiquetas_clases=("d = +1", "d = −1"),
        predictor=lambda Z, m=modelo: m.predecir(Z),
    )
    ruta = vz.guardar(fig, FIGURAS / "03_xor_frontera.png")
    reporte.figura(ruta, "Sea cual sea la recta, siempre queda al menos un patron del lado equivocado.")

    # Oscilacion de los pesos
    T = np.array([w[0] for w in modelo.trayectoria_pesos])
    fig, _ = vz.curvas_comparadas(
        {"w0 (inclinacion)": T[:, 0], "w1": T[:, 1], "w2": T[:, 2]},
        titulo="XOR: los pesos entran en un ciclo y no se estabilizan",
        etiqueta_y="valor del peso",
        etiqueta_x="numero de correccion",
        escala_log=False,
    )
    ruta = vz.guardar(fig, FIGURAS / "03_xor_oscilacion_pesos.png")
    reporte.figura(ruta, "Trayectoria de los tres pesos. El patron se repite ciclicamente: "
                         "el algoritmo deshace lo que acaba de aprender.")

    ciclo = pd.DataFrame(T[:12], columns=["w0", "w1", "w2"])
    ciclo.insert(0, "correccion", range(len(ciclo)))
    reporte.tabla(ciclo, "Primeras correcciones: el vector de pesos regresa a estados ya visitados",
                  nombre_csv="03_ciclo_pesos")

    # ------------------------------------------------------------------
    # 3. Busqueda exhaustiva
    # ------------------------------------------------------------------
    reporte.seccion("3. Busqueda exhaustiva: no es culpa del algoritmo")
    filas = []
    for nombre in ("AND", "OR", "XOR"):
        c = ds.compuerta(nombre, "bipolar")
        exactitud, w = busqueda_exhaustiva_recta(c, n=41, limite=3.0)
        filas.append({
            "compuerta": nombre,
            "mejor_exactitud": exactitud,
            "patrones_resueltos": f"{int(round(exactitud * 4))} de 4",
            "w0": round(float(w[0]), 3), "w1": round(float(w[1]), 3), "w2": round(float(w[2]), 3),
        })
    df_busqueda = pd.DataFrame(filas)
    print("\nBusqueda exhaustiva sobre 41^3 = 68921 rectas por compuerta:")
    print(df_busqueda.to_string(index=False))

    reporte.texto(
        "Se evaluaron **68 921 rectas** (rejilla de 41 valores por peso en [−3, 3]) sobre cada "
        "compuerta, midiendo cuantos de los cuatro patrones clasifica bien cada una. Si ninguna "
        "recta del espacio de pesos resuelve el XOR, el problema no esta en la regla de "
        "aprendizaje: esta en que la arquitectura solo sabe trazar rectas."
    )
    reporte.tabla(df_busqueda, "Mejor recta posible para cada compuerta",
                  nombre_csv="03_busqueda_rectas")
    reporte.texto(
        "AND y OR alcanzan el 100 %; el XOR se queda en el **75 %** --- tres patrones de cuatro. "
        "Ese 75 % es la cota superior absoluta de cualquier red unicapa sobre el XOR. Conviene "
        "senalar que el algoritmo perceptronico ni siquiera se estabiliza en esa cota: como "
        "sigue corrigiendo eternamente, la recta que tiene en un instante dado puede ser peor "
        "que la mejor posible (en las corridas de la seccion 2 oscila alrededor del 50 %). El "
        "algoritmo no esta minimizando ningun error --- solo reacciona al ultimo patron mal "
        "clasificado --- y esa es justamente la carencia que el ADALINE corrige con la regla "
        "Delta (experimento 05)."
    )

    # ------------------------------------------------------------------
    # 4. Salidas al problema
    # ------------------------------------------------------------------
    reporte.seccion("4. Como se resuelve entonces el XOR")
    red = red_xor()
    tabla = red.tabla_verdad()
    reporte.texto(
        "Hay dos caminos, y ambos consisten en **anadir una capa**, no en cambiar la regla de "
        "aprendizaje:"
    )
    reporte.lista([
        "**Componer neuronas a mano** (experimento 01): XOR = OR ∧ ¬AND, con tres neuronas de "
        "McCulloch-Pitts. Funciona, pero los pesos se disenan, no se aprenden.",
        "**Red multicapa** (`ICE-claseRN03.md`, seccion *Red Multicapa*): una capa oculta no "
        "lineal genera regiones convexas, y con tres capas se obtienen regiones arbitrarias. El "
        "precio es que la regla perceptronica ya no sirve --- no hay salida deseada para las "
        "neuronas ocultas --- y hace falta retropropagacion, que queda fuera del alcance de "
        "estos experimentos de red unicapa.",
    ])
    reporte.tabla(tabla, "La red compuesta de tres neuronas MCP si reproduce el XOR",
                  nombre_csv="03_xor_red_compuesta")
    reporte.texto(
        "La capa oculta transforma el problema: en el espacio (h_or, h_and) los cuatro patrones "
        "**si** son linealmente separables, y por eso la neurona de salida --- que sigue siendo "
        "un simple umbral lineal --- puede resolverlo. Esa es la funcion de las capas ocultas."
    )

    # Representacion interna
    h = red.evaluar(ds.compuerta("XOR").X)
    X_oculto = np.column_stack([h["h_or"], h["h_and"]])
    d_xor = 2 * ds.compuerta("XOR").d - 1
    fig, _ = vz.frontera_decision(
        np.array([-1.0, 1.0, -2.0]), X_oculto + np.random.default_rng(0).normal(0, 0.02, X_oculto.shape),
        d_xor,
        titulo="El XOR visto desde la capa oculta: ya es separable",
        etiquetas_ejes=("h_or", "h_and"),
        etiquetas_clases=("y = 1", "y = 0"),
        margen=0.5, anotar_patrones=False,
    )
    ruta = vz.guardar(fig, FIGURAS / "03_xor_espacio_oculto.png")
    reporte.figura(ruta, "Proyeccion de los cuatro patrones en el espacio de la capa oculta "
                         "(con un desplazamiento minimo para separar los puntos superpuestos). "
                         "Una sola recta basta.")

    # ------------------------------------------------------------------
    # 5. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("5. Conclusiones")
    reporte.lista([
        f"El perceptron simple **no converge** sobre el XOR: {len(modelo.trayectoria_pesos) - 1} "
        f"correcciones en {modelo.epocas_usadas} epocas sin cumplir nunca el criterio de parada.",
        "El maximo teorico para cualquier recta es del 75 % (verificado sobre 68 921 rectas), y "
        "el algoritmo ni siquiera se detiene ahi: oscila indefinidamente entre soluciones "
        "parciales.",
        "El fallo es **estructural**: una red unicapa solo genera una frontera lineal.",
        "Anadir una capa oculta transforma el espacio de representacion y devuelve el problema al "
        "terreno linealmente separable.",
        "Esta limitacion, publicada por Minsky y Papert en 1969, es la que detuvo la "
        "investigacion en redes neuronales durante mas de una decada.",
    ])

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
