"""
Experimento 02 --- Perceptron simple: aprendizaje de las compuertas AND y OR.

A diferencia del experimento 01, aqui los pesos **no se disenan**: la red los
descubre sola a partir de los cuatro pares de entrenamiento, aplicando la regla
perceptronica de `ICE-claseRN03.md` (pasos 0 a 5).

Contenido
---------
1. Arquitectura de la red unicapa con neurona de inclinacion.
2. Traza completa del algoritmo sobre el AND, patron por patron.
3. Convergencia sobre AND y OR: epocas, correcciones, pesos finales, frontera.
4. Evolucion de la frontera de decision durante el aprendizaje.
5. Estudio del efecto de la razon de aprendizaje `a`.
6. Estudio del efecto del punto de indeterminacion `theta` sobre el margen.
7. Efecto de la codificacion (binaria frente a bipolar).
8. Verificacion de que la solucion no es unica: el haz de infinitas soluciones.

Ejecucion
---------
    python experimentos/exp02_perceptron_and_or.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.perceptron import PerceptronSimple
from ce_rna.reportes import Reporte, encabezado_experimento


def entrenar_compuerta(nombre, codificacion="bipolar", a=1.0, theta=0.0,
                       inicializacion="ceros", semilla=0, max_epocas=100):
    """Entrena un perceptron sobre una compuerta y devuelve (modelo, conjunto)."""
    conjunto = ds.compuerta(nombre, codificacion)
    modelo = PerceptronSimple(
        n_entradas=2, n_salidas=1, razon_aprendizaje=a, theta=theta,
        inicializacion=inicializacion, semilla=semilla, max_epocas=max_epocas,
    )
    modelo.entrenar(conjunto.X, conjunto.d, registrar_traza=True)
    return modelo, conjunto


def main() -> None:
    encabezado_experimento("Experimento 02 - Perceptron simple: compuertas AND y OR")

    reporte = Reporte(
        titulo="Experimento 02 --- Perceptron simple: aprendizaje de AND y OR",
        nombre_archivo="02_perceptron_and_or",
        resumen=(
            "El perceptron de Rosenblatt (1958) es la primera red **que aprende**: en lugar de "
            "fijar los pesos a mano como en McCulloch-Pitts, los ajusta a partir de ejemplos "
            "mediante la regla perceptronica. Este experimento entrena una red unicapa de una "
            "sola neurona de salida sobre las compuertas AND y OR, audita el algoritmo paso a "
            "paso y estudia como afectan la razon de aprendizaje, el punto de indeterminacion, "
            "la inicializacion y la codificacion de los datos."
        ),
    )

    # ------------------------------------------------------------------
    # 1. Arquitectura y algoritmo
    # ------------------------------------------------------------------
    reporte.seccion("1. Arquitectura y regla de aprendizaje")
    reporte.texto(
        "La red es **unicapa**: m0 = 2 entradas mas la *neurona de inclinacion* x0 = 1 "
        "(que sustituye al umbral) y m1 = 1 neurona de salida no lineal con activacion "
        "escalon bipolar."
    )
    reporte.formula(r"y^{(in)}(n) = \sum_{i=0}^{m_0} w_i(n)\,x_i(n), \qquad "
                    r"y(n) = \varphi\left(y^{(in)}(n)\right) = "
                    r"\begin{cases} +1 & y^{(in)} > \theta \\ 0 & |y^{(in)}| \leq \theta \\ "
                    r"-1 & y^{(in)} < -\theta \end{cases}")
    reporte.texto("**Algoritmo perceptronico** (pasos 0 a 5 de `ICE-claseRN03.md`):")
    reporte.lista([
        "**Paso 0**: inicializar las sinapsis (w = 0 o aleatorias) y elegir 0 < a < 1.",
        "**Paso 1**: repetir mientras la condicion de parada del paso 5 sea falsa.",
        "**Paso 2**: para cada par de entrenamiento (x_i(n), d(n)).",
        "**Paso 3**: calcular el potencial y la salida.",
        "**Paso 4**: si y ≠ d, corregir `w_i(n+1) = w_i(n) + a·d·x_i(n)`; si no, no tocar nada.",
        "**Paso 5**: si los pesos no cambiaron en toda una epoca, parar.",
    ])
    reporte.texto(
        "Notese que la correccion usa la **salida deseada d**, no el error d − y: es la forma "
        "clasica de Rosenblatt que emplea la clase. Como la correccion solo se aplica cuando "
        "hay error, y en codificacion bipolar el error vale d − y = 2d cuando y = −d, ambas "
        "formulaciones difieren unicamente en un factor constante que absorbe la razon de "
        "aprendizaje."
    )

    modelo_and, conj_and = entrenar_compuerta("AND")
    fig, _ = vz.dibujar_red(
        [["x0=1", "x1", "x2"], ["y"]],
        titulo="Perceptron simple entrenado sobre AND (pesos finales)",
        pesos=[modelo_and.W],
        etiquetas_capas=["Capa 0 (entradas + inclinacion)", "Capa 1 (salida)"],
        nota="La unidad x0 = 1 es la neurona de inclinacion: su peso w0 hace las veces de umbral.",
    )
    ruta = vz.guardar(fig, FIGURAS / "02_arquitectura_perceptron.png")
    reporte.figura(ruta, "Red unicapa con neurona de inclinacion y los pesos aprendidos para el AND.")

    # ------------------------------------------------------------------
    # 2. Traza del algoritmo
    # ------------------------------------------------------------------
    reporte.seccion("2. Traza paso a paso sobre el AND")
    print("\nTraza del algoritmo perceptronico sobre AND (a = 1, pesos iniciales nulos):")
    print(modelo_and.traza_df().to_string(index=False))
    reporte.texto(
        "Pesos iniciales nulos, a = 1, codificacion bipolar. Cada fila es una presentacion de "
        "patron: se muestran el potencial `y_in`, la respuesta `y`, la deseada `d`, si hubo "
        "correccion y los pesos **despues** de aplicarla."
    )
    reporte.tabla(modelo_and.traza_df(), "Traza del entrenamiento del AND",
                  nombre_csv="02_traza_and")
    reporte.texto(
        "El primer patron (−1,−1) produce y_in = 0 y, por el criterio y_in ≥ 0 → +1, la red "
        "responde +1 cuando deberia responder −1. La correccion w ← w + a·d·x con d = −1 deja "
        f"`w = {np.array2string(modelo_and.W[0], precision=0)}`, que ya resuelve los cuatro "
        "patrones; la segunda epoca transcurre sin cambios y el paso 5 detiene el algoritmo."
    )

    # Traza con inicializacion aleatoria, mas larga e ilustrativa
    modelo_rand, _ = entrenar_compuerta("AND", a=0.4, inicializacion="aleatoria", semilla=7)
    reporte.texto(
        f"Con inicializacion aleatoria y a = 0.4 el recorrido es mas largo "
        f"({modelo_rand.epocas_usadas} epocas, {len(modelo_rand.trayectoria_pesos) - 1} correcciones) "
        f"y termina en pesos distintos, `w = {np.array2string(modelo_rand.W[0], precision=3)}`, "
        "que sin embargo clasifican igual de bien: la solucion no es unica."
    )
    reporte.tabla(modelo_rand.traza_df().head(12),
                  "Primeras 12 presentaciones con inicializacion aleatoria (a = 0.4)",
                  nombre_csv="02_traza_and_aleatoria")

    # ------------------------------------------------------------------
    # 3. Convergencia sobre AND y OR
    # ------------------------------------------------------------------
    reporte.seccion("3. Convergencia sobre AND y OR")
    filas = []
    for nombre in ("AND", "OR"):
        modelo, conjunto = entrenar_compuerta(nombre)
        y = modelo.predecir(conjunto.X)
        filas.append({
            "compuerta": nombre,
            "epocas": modelo.epocas_usadas,
            "correcciones": len(modelo.trayectoria_pesos) - 1,
            "convergio": "si" if modelo.convergio else "no",
            "w0": modelo.W[0, 0], "w1": modelo.W[0, 1], "w2": modelo.W[0, 2],
            "exactitud": mt.exactitud(y, conjunto.d),
            "margen": mt.margen_geometrico(modelo.W[0], conjunto.X, conjunto.d),
        })
        print(f"\n{nombre}: {modelo.resumen()}")

        fig, _ = vz.frontera_decision(
            modelo.W[0], conjunto.X, conjunto.d,
            titulo=f"Perceptron sobre {nombre}: frontera aprendida",
            etiquetas_clases=("d = +1", "d = −1"),
            predictor=lambda Z, m=modelo: m.predecir(Z),
        )
        ruta = vz.guardar(fig, FIGURAS / f"02_frontera_{nombre.lower()}.png")
        reporte.figura(ruta, f"Region de decision aprendida para {nombre}. "
                             f"Recta: {modelo.W[0,0]:.0f} + {modelo.W[0,1]:.0f}·x1 + {modelo.W[0,2]:.0f}·x2 = 0.")

        fig, _ = vz.evolucion_fronteras(
            [w[0] for w in modelo_rand.trayectoria_pesos] if nombre == "AND"
            else [w[0] for w in modelo.trayectoria_pesos],
            conjunto.X, conjunto.d,
            titulo=f"Evolucion de la frontera durante el aprendizaje ({nombre})",
            etiquetas_clases=("d = +1", "d = −1"),
        )
        ruta = vz.guardar(fig, FIGURAS / f"02_evolucion_{nombre.lower()}.png")
        detalle = ("corrida con inicializacion aleatoria y a = 0.4, que hace mas correcciones "
                   "y deja ver el recorrido") if nombre == "AND" else "corrida desde pesos nulos con a = 1"
        reporte.figura(ruta, f"Cada correccion rota y desplaza el hiperplano ({detalle}); "
                             "la recta solida es la frontera final.")

    df_conv = pd.DataFrame(filas)
    print("\nResumen de convergencia:")
    print(df_conv.to_string(index=False))
    reporte.tabla(df_conv, "Convergencia con pesos iniciales nulos y a = 1",
                  nombre_csv="02_convergencia_and_or")
    reporte.texto(
        "El **margen geometrico** es la distancia minima de un patron a la frontera. Un margen "
        "positivo certifica que la separacion es correcta; su magnitud indica cuanta holgura "
        "tiene la solucion frente al ruido."
    )

    # ------------------------------------------------------------------
    # 4. Efecto de la razon de aprendizaje
    # ------------------------------------------------------------------
    reporte.seccion("4. Efecto de la razon de aprendizaje")
    filas = []
    for a in (0.1, 0.25, 0.5, 0.75, 1.0):
        for inic, semilla in (("ceros", 0), ("aleatoria", 3)):
            modelo, conjunto = entrenar_compuerta("AND", a=a, inicializacion=inic, semilla=semilla)
            filas.append({
                "a": a,
                "inicializacion": inic,
                "epocas": modelo.epocas_usadas,
                "correcciones": len(modelo.trayectoria_pesos) - 1,
                "w0": round(float(modelo.W[0, 0]), 4),
                "w1": round(float(modelo.W[0, 1]), 4),
                "w2": round(float(modelo.W[0, 2]), 4),
                "margen": round(mt.margen_geometrico(modelo.W[0], conjunto.X, conjunto.d), 4),
            })
    df_a = pd.DataFrame(filas)
    print("\nEfecto de la razon de aprendizaje:")
    print(df_a.to_string(index=False))
    reporte.tabla(df_a, "Razon de aprendizaje frente a coste de convergencia (compuerta AND)",
                  nombre_csv="02_razon_aprendizaje")
    reporte.texto(
        "Con **pesos iniciales nulos la razon de aprendizaje es irrelevante**: todos los pesos "
        "son multiplos de a, de modo que la frontera w·x = 0 --- y por tanto la sucesion de "
        "decisiones --- es identica para cualquier a. El numero de epocas solo cambia cuando "
        "los pesos iniciales no son nulos, porque entonces a controla el peso relativo de cada "
        "correccion frente al estado de partida."
    )

    # ------------------------------------------------------------------
    # 5. Punto de indeterminacion
    # ------------------------------------------------------------------
    reporte.seccion("5. Efecto del punto de indeterminacion θ")
    filas = []
    for theta in (0.0, 0.5, 1.0, 2.0):
        modelo, conjunto = entrenar_compuerta("AND", theta=theta, max_epocas=200)
        filas.append({
            "theta": theta,
            "epocas": modelo.epocas_usadas,
            "correcciones": len(modelo.trayectoria_pesos) - 1,
            "convergio": "si" if modelo.convergio else "no",
            "w0": float(modelo.W[0, 0]), "w1": float(modelo.W[0, 1]), "w2": float(modelo.W[0, 2]),
            "margen": round(mt.margen_geometrico(modelo.W[0], conjunto.X, conjunto.d), 4),
            "|w_in| minimo": round(float(np.min(np.abs(modelo.potencial(conjunto.X)))), 4),
        })
    df_theta = pd.DataFrame(filas)
    print("\nEfecto del punto de indeterminacion:")
    print(df_theta.to_string(index=False))
    reporte.tabla(df_theta, "El punto neutro obliga a separar con margen (compuerta AND)",
                  nombre_csv="02_theta")
    reporte.texto(
        "Una salida dentro de la banda |y_in| ≤ θ significa *\"la neurona no sabe que "
        "responder\"* y se cuenta como error, de modo que el algoritmo sigue corrigiendo hasta "
        "que **todos** los patrones queden fuera de la banda. El resultado es una frontera con "
        "mas holgura: la columna `|w_in| minimo` crece con θ. Es el mismo principio que, llevado "
        "al limite, da lugar a los clasificadores de margen maximo."
    )

    # ------------------------------------------------------------------
    # 6. Codificacion binaria frente a bipolar
    # ------------------------------------------------------------------
    reporte.seccion("6. Codificacion binaria frente a bipolar")
    filas = []
    for codificacion in ("binaria", "bipolar"):
        for nombre in ("AND", "OR"):
            conjunto = ds.compuerta(nombre, codificacion)
            objetivo = conjunto.d if codificacion == "bipolar" else 2 * conjunto.d - 1
            modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=100)
            modelo.entrenar(conjunto.X, objetivo)
            filas.append({
                "codificacion": codificacion,
                "compuerta": nombre,
                "epocas": modelo.epocas_usadas,
                "correcciones": len(modelo.trayectoria_pesos) - 1,
                "exactitud": mt.exactitud(modelo.predecir(conjunto.X), objetivo),
            })
    df_cod = pd.DataFrame(filas)
    print("\nCodificacion binaria frente a bipolar:")
    print(df_cod.to_string(index=False))
    reporte.tabla(df_cod, "Ambas codificaciones convergen, pero no al mismo coste",
                  nombre_csv="02_codificacion")
    reporte.texto(
        "Con entradas binarias, un patron con x_i = 0 **no puede modificar** el peso w_i, porque "
        "la correccion es proporcional a la entrada: los patrones apagados solo ajustan el sesgo. "
        "Con codificacion bipolar cada presentacion informa a todos los pesos. El perceptron "
        "converge igualmente en ambos casos --- su teorema de convergencia no depende de la "
        "codificacion --- pero la bipolar suele necesitar menos correcciones, y para la regla de "
        "Hebb (experimento 07) la diferencia es entre funcionar y no funcionar."
    )

    # ------------------------------------------------------------------
    # 7. Infinitas soluciones
    # ------------------------------------------------------------------
    reporte.seccion("7. La solucion no es unica: infinitas fronteras validas")
    conjunto = ds.nubes_separables(n_por_clase=35, separacion=3.2, dispersion=0.75, semilla=11)
    soluciones = []
    for semilla in range(60):
        modelo = PerceptronSimple(2, 1, razon_aprendizaje=0.3, inicializacion="aleatoria",
                                  semilla=semilla, max_epocas=200)
        modelo.entrenar(conjunto.X, conjunto.d)
        if mt.exactitud(modelo.predecir(conjunto.X), conjunto.d) == 1.0:
            soluciones.append(modelo.W[0])
    print(f"\nSoluciones validas encontradas con 60 inicializaciones distintas: {len(soluciones)}")

    fig, _ = vz.haz_de_soluciones(
        soluciones, conjunto.X, conjunto.d,
        titulo="Infinitas soluciones para un problema linealmente separable",
    )
    ruta = vz.guardar(fig, FIGURAS / "02_infinitas_soluciones.png")
    reporte.figura(ruta, f"{len(soluciones)} fronteras obtenidas con {60} inicializaciones aleatorias "
                         "distintas. Todas clasifican el 100% de los patrones.")
    reporte.texto(
        "*\"Es bastante evidente que si un problema es linealmente separable, existen infinitos "
        "pesos sinapticos que serviran para solucionar el problema... o no existe ninguna "
        "solucion, o existen infinitas\"* (`ICE-claseRN03.md`). Basta multiplicar w por una "
        "constante positiva para obtener el mismo hiperplano con pesos distintos; y ademas hay "
        "infinitos hiperplanos que separan correctamente. El perceptron se detiene en **el "
        "primero** que encuentra, que depende de la inicializacion y del orden de presentacion: "
        "no busca el mejor, solo uno valido. Esa es una limitacion real frente al ADALINE, que "
        "si tiene un criterio de optimalidad (el error cuadratico medio)."
    )

    margenes = [mt.margen_geometrico(w, conjunto.X, conjunto.d) for w in soluciones]
    df_margenes = pd.DataFrame({
        "estadistico": ["minimo", "mediana", "maximo"],
        "margen_geometrico": [np.min(margenes), np.median(margenes), np.max(margenes)],
    })
    reporte.tabla(df_margenes, "Dispersion del margen entre las soluciones halladas",
                  nombre_csv="02_margenes_soluciones")

    # ------------------------------------------------------------------
    # 8. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("8. Conclusiones")
    reporte.lista([
        "El perceptron aprende AND y OR **desde pesos nulos en 2 epocas** y una sola correccion "
        "por compuerta: ambos problemas son linealmente separables y el teorema de convergencia "
        "garantiza un numero finito de pasos.",
        "La traza confirma que el algoritmo solo toca los pesos cuando se equivoca; cuando "
        "acierta, w(n+1) = w(n).",
        "Con pesos iniciales nulos la razon de aprendizaje **no altera la frontera**, solo la "
        "escala de los pesos. Con pesos aleatorios si influye en cuantas correcciones hacen falta.",
        "El punto de indeterminacion θ actua como exigencia de margen: fuerza soluciones mas "
        "holgadas a cambio de mas correcciones.",
        "La solucion hallada no es unica ni optima: con 60 inicializaciones se obtienen 60 "
        "fronteras validas distintas, con margenes que van de "
        f"{np.min(margenes):.2f} a {np.max(margenes):.2f}.",
    ])

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
