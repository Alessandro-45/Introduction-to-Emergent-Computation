"""
Experimento 06 --- Perceptron frente a ADALINE sobre las mismas compuertas.

Ambos modelos comparten arquitectura (una capa, una neurona, entradas mas
sesgo); lo unico que cambia es **que se usa para aprender**: el perceptron
corrige a partir de la salida umbralizada, el ADALINE a partir de la salida
lineal.  `ICE-claseRN04.md` enumera cuatro diferencias; este experimento las
mide.

Contenido
---------
1. Las dos reglas, lado a lado.
2. Fronteras que produce cada uno sobre AND y OR.
3. Margen geometrico y robustez frente a entradas perturbadas.
4. Numero de actualizaciones y criterio de parada.
5. El ADALINE sobre el XOR: tambien falla, pero de otra manera.

Ejecucion
---------
    python experimentos/exp06_perceptron_vs_adaline.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.adaline import Adaline, solucion_minimos_cuadrados
from ce_rna.perceptron import PerceptronSimple
from ce_rna.reportes import Reporte, encabezado_experimento


def robustez(w, conjunto, sigma, n_ensayos=2000, semilla=0) -> float:
    """Exactitud al perturbar las entradas con ruido gaussiano de desviacion sigma."""
    rng = np.random.default_rng(semilla)
    w = np.asarray(w, dtype=float).ravel()
    aciertos = 0
    for _ in range(n_ensayos):
        X = conjunto.X + rng.normal(0.0, sigma, conjunto.X.shape)
        Xe = np.hstack([np.ones((X.shape[0], 1)), X])
        y = np.where(Xe @ w >= 0, 1.0, -1.0)
        aciertos += int(np.sum(y == conjunto.d))
    return aciertos / (n_ensayos * conjunto.n_patrones)


def main() -> None:
    encabezado_experimento("Experimento 06 - Perceptron frente a ADALINE")

    reporte = Reporte(
        titulo="Experimento 06 --- Perceptron frente a ADALINE sobre las mismas compuertas",
        nombre_archivo="06_perceptron_vs_adaline",
        resumen=(
            "Misma arquitectura, misma tarea, dos reglas de aprendizaje. El perceptron deja de "
            "corregir en cuanto acierta; el ADALINE sigue reduciendo el error aunque ya clasifique "
            "bien. Este experimento cuantifica que consecuencias tiene esa diferencia sobre la "
            "calidad de la frontera obtenida."
        ),
    )

    # ------------------------------------------------------------------
    # 1. Las dos reglas
    # ------------------------------------------------------------------
    reporte.seccion("1. Las dos reglas, lado a lado")
    reporte.formula(r"\text{Perceptron:}\quad \Delta w_i = a\,d\,x_i \;\;\text{ solo si } y \neq d "
                    r"\qquad\qquad \text{ADALINE:}\quad \Delta w_j = \gamma\,(d - y)\,x_j \;\;\text{ siempre}")
    reporte.texto(
        "En el perceptron, `y` es la salida **umbralizada** y la correccion es todo o nada. En el "
        "ADALINE, `y` es la salida **lineal** y la correccion es proporcional al error cometido. "
        "De ahi la observacion de la clase: *\"en el ADALINE existe una medida de cuanto se ha "
        "equivocado la red; en el PERCEPTRON solo se determina si se ha equivocado o no\"*."
    )

    # ------------------------------------------------------------------
    # 2 y 3. Fronteras, margenes y robustez
    # ------------------------------------------------------------------
    reporte.seccion("2. Fronteras, margenes y robustez")
    filas = []
    for nombre in ("AND", "OR"):
        conjunto = ds.compuerta(nombre, "bipolar")

        perceptron = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=100)
        perceptron.entrenar(conjunto.X, conjunto.d)
        w_perc = perceptron.W[0]

        adaline = Adaline(2, razon_aprendizaje=0.05, max_epocas=3000, tolerancia=1e-10,
                          inicializacion="aleatoria", semilla=3)
        adaline.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
        w_ada = adaline.w

        for etiqueta, w, actualizaciones in (
            ("perceptron", w_perc, len(perceptron.trayectoria_pesos) - 1),
            ("adaline", w_ada, adaline.epocas_usadas * conjunto.n_patrones),
        ):
            Xe = np.hstack([np.ones((conjunto.X.shape[0], 1)), conjunto.X])
            y_lineal = Xe @ w
            y_clase = np.where(y_lineal >= 0, 1.0, -1.0)
            filas.append({
                "compuerta": nombre,
                "modelo": etiqueta,
                "actualizaciones": actualizaciones,
                "exactitud": mt.exactitud(y_clase, conjunto.d),
                "margen": round(mt.margen_geometrico(w, conjunto.X, conjunto.d), 4),
                "ecm": round(mt.ecm(y_lineal, conjunto.d), 4),
                "robustez σ=0.4": round(robustez(w, conjunto, 0.4, semilla=1), 4),
                "robustez σ=0.8": round(robustez(w, conjunto, 0.8, semilla=2), 4),
                "w": np.array2string(w, precision=3),
            })

        fig, ejes = vz.plt.subplots(1, 2, figsize=(11.2, 4.8))
        vz.frontera_decision(w_perc, conjunto.X, conjunto.d,
                             titulo=f"Perceptron sobre {nombre}",
                             etiquetas_clases=("d = +1", "d = −1"), ax=ejes[0])
        vz.frontera_decision(w_ada, conjunto.X, conjunto.d,
                             titulo=f"ADALINE sobre {nombre}",
                             etiquetas_clases=("d = +1", "d = −1"), ax=ejes[1])
        fig.tight_layout()
        ruta = vz.guardar(fig, FIGURAS / f"06_comparacion_{nombre.lower()}.png")
        reporte.figura(ruta, f"Fronteras obtenidas por cada regla sobre la compuerta {nombre}.")

    df = pd.DataFrame(filas)
    print("\nComparacion perceptron / ADALINE:")
    print(df.to_string(index=False))
    reporte.tabla(df, "Comparacion cuantitativa sobre AND y OR", nombre_csv="06_comparacion")
    reporte.texto(
        "Las columnas de robustez miden la exactitud cuando las entradas se perturban con ruido "
        "gaussiano (2000 ensayos por nivel). Ambos modelos clasifican perfectamente los cuatro "
        "patrones y, sobre estas compuertas, **acaban proponiendo practicamente la misma "
        "frontera**: la solucion de minimos cuadrados del AND bipolar es exactamente "
        "w = (−0.5, 0.5, 0.5), que es proporcional a la (−1, 1, 1) del perceptron. Las pequenas "
        "diferencias de margen que muestra la tabla no son una propiedad del ADALINE, sino el "
        "residuo de haberse detenido por tolerancia antes de alcanzar el minimo exacto."
    )
    reporte.texto(
        "Es un resultado que conviene registrar con claridad, porque contradice la intuicion "
        "habitual de que *\"el ADALINE coloca mejor la frontera\"*. Sobre cuatro patrones "
        "simetricos las dos reglas coinciden. Para ver una diferencia real hay que ir a un "
        "problema con muchos patrones y dispersion, como el de la seccion siguiente."
    )

    # ------------------------------------------------------------------
    # 2 bis. Donde si difieren: nubes de puntos
    # ------------------------------------------------------------------
    reporte.seccion("3. Donde las dos reglas si difieren: nubes de puntos")
    nubes = ds.nubes_separables(n_por_clase=30, separacion=3.0, dispersion=0.8, semilla=11)
    filas_nubes = []
    for semilla in range(5):
        p_nube = PerceptronSimple(2, 1, razon_aprendizaje=0.3, inicializacion="aleatoria",
                                  semilla=semilla, max_epocas=300)
        p_nube.entrenar(nubes.X, nubes.d)
        filas_nubes.append({
            "modelo": f"perceptron (semilla {semilla})",
            "exactitud": mt.exactitud(p_nube.predecir(nubes.X), nubes.d),
            "margen": round(mt.margen_geometrico(p_nube.W[0], nubes.X, nubes.d), 4),
            "ecm": round(mt.ecm(np.hstack([np.ones((nubes.X.shape[0], 1)), nubes.X]) @ p_nube.W[0], nubes.d), 4),
        })
    w_lms_nubes = solucion_minimos_cuadrados(nubes.X, nubes.d)
    a_nube = Adaline(2, razon_aprendizaje=0.02, max_epocas=5000, tolerancia=1e-12,
                     inicializacion="aleatoria", semilla=3)
    a_nube.entrenar(nubes.X, nubes.d, registrar_pasos=False)
    for etiqueta, w in (("adaline", a_nube.w), ("minimos cuadrados (exacto)", w_lms_nubes)):
        Xe = np.hstack([np.ones((nubes.X.shape[0], 1)), nubes.X])
        filas_nubes.append({
            "modelo": etiqueta,
            "exactitud": mt.exactitud(np.where(Xe @ w >= 0, 1.0, -1.0), nubes.d),
            "margen": round(mt.margen_geometrico(w, nubes.X, nubes.d), 4),
            "ecm": round(mt.ecm(Xe @ w, nubes.d), 4),
        })
    df_nubes = pd.DataFrame(filas_nubes)
    print("\nNubes de puntos (60 patrones):")
    print(df_nubes.to_string(index=False))
    reporte.tabla(df_nubes, "Perceptron y ADALINE sobre 60 patrones dispersos pero separables",
                  nombre_csv="06_nubes")

    fig, ejes = vz.plt.subplots(1, 2, figsize=(11.6, 4.9))
    p_ref = PerceptronSimple(2, 1, razon_aprendizaje=0.3, inicializacion="aleatoria",
                             semilla=2, max_epocas=300)
    p_ref.entrenar(nubes.X, nubes.d)
    vz.frontera_decision(p_ref.W[0], nubes.X, nubes.d,
                         titulo="Perceptron: separa los 60 patrones",
                         anotar_patrones=False, ax=ejes[0])
    vz.frontera_decision(w_lms_nubes, nubes.X, nubes.d,
                         titulo="Minimos cuadrados: minimiza el ECM",
                         anotar_patrones=False, ax=ejes[1])
    fig.tight_layout()
    ruta = vz.guardar(fig, FIGURAS / "06_nubes_comparacion.png")
    reporte.figura(ruta, "Dos objetivos distintos sobre los mismos datos. La recta de la derecha "
                         "tiene menor error cuadratico; la de la izquierda clasifica mejor.")

    exactitud_lms = df_nubes.loc[df_nubes["modelo"] == "minimos cuadrados (exacto)", "exactitud"].iloc[0]
    reporte.texto(
        f"Aqui la diferencia aparece, y en la direccion contraria a la esperada: la solucion de "
        f"minimos cuadrados --- el optimo exacto al que tiende el ADALINE --- clasifica bien el "
        f"{exactitud_lms:.1%} de los patrones y tiene **margen negativo**, mientras que el "
        "perceptron los separa todos. No es un fallo de implementacion: es que **los dos modelos "
        "optimizan cosas distintas**. Minimizar el error cuadratico penaliza a los patrones muy "
        "alejados de la frontera --- aunque esten bien clasificados --- y puede inclinar la recta "
        "hasta cruzar a un patron correcto con tal de reducir esa penalizacion. El perceptron, que "
        "solo mira los errores de clasificacion, es inmune a ese efecto."
    )
    reporte.texto(
        "Un matiz que la tabla deja ver: el ADALINE entrenado por descenso del gradiente si "
        "clasifica el 100 %, porque se detiene por tolerancia **antes** de llegar al optimo "
        "exacto. Detenerse pronto actua aqui como una regularizacion accidental --- el modelo se "
        "queda a medio camino entre el punto de partida y una solucion peor para clasificar. No "
        "es una virtud del metodo, es una casualidad afortunada del criterio de parada, y "
        "conviene no confundir una cosa con la otra."
    )
    reporte.texto(
        "La leccion practica: el ADALINE es la herramienta adecuada cuando la salida deseada es "
        "una **magnitud real** que se quiere aproximar (experimento 05); para clasificar pura y "
        "simplemente, el error cuadratico es un objetivo sustituto, no el objetivo real."
    )

    # ------------------------------------------------------------------
    # 4. Criterio de parada
    # ------------------------------------------------------------------
    reporte.seccion("4. Dos criterios de parada distintos")
    reporte.lista([
        "**Perceptron** (paso 5): *\"si los pesos sinapticos no cambian para cada patron de "
        "entrenamiento durante la ultima vez que se realizo el paso 2, parar\"*. Es un criterio "
        "**exacto y alcanzable**: cuando no hay errores, no hay correcciones, y el algoritmo se "
        "detiene solo. Si el problema no es separable, no se detiene nunca.",
        "**ADALINE**: el error cuadratico tiende a su minimo de forma asintotica, pero rara vez "
        "lo alcanza exactamente en aritmetica finita. Hace falta un criterio **aproximado**: "
        "detenerse cuando el mayor cambio de peso cae por debajo de una tolerancia, o cuando se "
        "agota un numero de epocas. A cambio, el algoritmo siempre termina, tambien en problemas "
        "no separables.",
    ])

    # ------------------------------------------------------------------
    # 5. ADALINE sobre el XOR
    # ------------------------------------------------------------------
    reporte.seccion("5. El ADALINE sobre el XOR: tambien falla, pero de otra manera")
    conjunto_xor = ds.compuerta("XOR", "bipolar")
    adaline_xor = Adaline(2, razon_aprendizaje=0.05, max_epocas=2000, tolerancia=1e-10,
                          inicializacion="aleatoria", semilla=3)
    adaline_xor.entrenar(conjunto_xor.X, conjunto_xor.d, registrar_pasos=False)
    w_lms = solucion_minimos_cuadrados(conjunto_xor.X, conjunto_xor.d)

    print("\nADALINE sobre XOR:")
    print(adaline_xor.resumen())
    print(f"  solucion de minimos cuadrados: {np.round(w_lms, 6)}")
    print(f"  ECM final: {adaline_xor.ecm(conjunto_xor.X, conjunto_xor.d):.6f}")

    df_xor = pd.DataFrame({
        "patron": [f"({int(a)},{int(b)})" for a, b in conjunto_xor.X],
        "d": conjunto_xor.d,
        "y (lineal)": adaline_xor.predecir(conjunto_xor.X),
        "y (umbralizada)": adaline_xor.predecir_clase(conjunto_xor.X),
        "error": conjunto_xor.d - adaline_xor.predecir(conjunto_xor.X),
    })
    reporte.tabla(df_xor, "Salida del ADALINE sobre los cuatro patrones del XOR",
                  nombre_csv="06_adaline_xor")
    reporte.texto(
        f"El ADALINE **si converge** sobre el XOR --- en {adaline_xor.epocas_usadas} epocas --- "
        "pero converge a la mejor aproximacion lineal posible, que es la solucion trivial "
        f"w = {np.array2string(np.round(adaline_xor.w, 4))}: la red predice 0 para los cuatro "
        f"patrones, con un ECM de {adaline_xor.ecm(conjunto_xor.X, conjunto_xor.d):.4f}. Es el "
        "comportamiento esperable de una regresion lineal sobre datos sin componente lineal: "
        "todas las correlaciones entrada-salida se cancelan."
    )
    reporte.texto(
        "La diferencia de comportamiento es instructiva. Ante un problema imposible, el "
        "perceptron **oscila indefinidamente** y el ADALINE **se detiene tranquilamente en la "
        "mejor solucion mala**. El ADALINE nunca avisa de que el problema no es soluble: hay que "
        "mirar el error residual para darse cuenta. Ambos comparten la misma limitacion de fondo "
        "--- una capa, una frontera lineal --- que solo se supera anadiendo capas."
    )

    # ------------------------------------------------------------------
    # 6. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("6. Conclusiones")
    reporte.lista([
        "Sobre las compuertas AND y OR ambas reglas convergen a **la misma frontera** (los pesos "
        "del perceptron son un multiplo exacto de la solucion de minimos cuadrados): con cuatro "
        "patrones simetricos no hay diferencia que medir.",
        "Sobre 60 patrones dispersos si la hay, y favorece al perceptron en clasificacion: la "
        "solucion de minimos cuadrados deja patrones mal clasificados que el perceptron separa "
        "correctamente, porque minimizar el error cuadratico no es lo mismo que separar clases.",
        "El perceptron tiene un criterio de parada exacto pero no termina si el problema no es "
        "separable; el ADALINE siempre termina, pero puede terminar en una solucion inutil sin "
        "senalarlo.",
        "El ADALINE produce salidas **reales**, lo que le permite abordar problemas de "
        "aproximacion de funciones (experimento 05) y no solo de clasificacion.",
        "Ninguno de los dos supera la barrera de la separabilidad lineal: es una limitacion de la "
        "arquitectura unicapa, no de la regla de aprendizaje.",
    ])

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
