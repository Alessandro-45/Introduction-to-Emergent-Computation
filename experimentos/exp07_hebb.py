"""
Experimento 07 --- Regla de Hebb: aprendizaje de una sola pasada.

`ICE-claseRN03.md` presenta la regla de Hebb (1949) como *"la mas antigua y la
mas famosa de las reglas de aprendizaje"*, con base biologica directa: si dos
neuronas a ambos lados de la sinapsis se activan simultaneamente, la conexion se
refuerza; si lo hacen asincronicamente, se debilita.

    Delta w_ji = a * d_j(n) * x_i(n)

Contenido
---------
1. La regla y su version supervisada.
2. AND y OR en codificacion **binaria**: la regla falla.
3. AND y OR en codificacion **bipolar**: la regla funciona.
4. Por que la codificacion decide el resultado.
5. Hebb como memoria asociativa: las letras X y O.
6. Comparacion con la regla perceptronica.

Ejecucion
---------
    python experimentos/exp07_hebb.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.hebb import RedHebb
from ce_rna.perceptron import PerceptronSimple
from ce_rna.reportes import Reporte, encabezado_experimento


def main() -> None:
    encabezado_experimento("Experimento 07 - Regla de Hebb")

    reporte = Reporte(
        titulo="Experimento 07 --- Regla de Hebb: aprendizaje de una sola pasada",
        nombre_archivo="07_regla_hebb",
        resumen=(
            "La regla de Hebb no comprueba errores, no itera y no tiene criterio de parada: "
            "presenta cada patron una vez y acumula el producto entrada x salida deseada. Es el "
            "aprendizaje mas barato posible, y el experimento delimita exactamente hasta donde "
            "llega --- incluido un caso en el que **falla**, que es el que mejor explica por que "
            "hizo falta inventar la regla perceptronica."
        ),
    )

    # ------------------------------------------------------------------
    # 1. La regla
    # ------------------------------------------------------------------
    reporte.seccion("1. La regla de Hebb")
    reporte.formula(r"\Delta w_{ji} = a\, y_j(n)\, x_i(n), \qquad a > 0")
    reporte.texto(
        "En su version **supervisada** para redes unicapa se sustituye la salida producida y_j "
        "por la salida deseada d_j, de modo que el peso final tras presentar los N patrones es"
    )
    reporte.formula(r"w_{ji} = a \sum_{n=1}^{N} d_j(n)\, x_i(n)")
    reporte.texto(
        "es decir, la **correlacion** entre la entrada i y la salida deseada j acumulada sobre "
        "todo el conjunto. No hay epocas, no hay errores y no hay criterio de parada: una sola "
        "pasada y el aprendizaje ha terminado. La razon de aprendizaje `a` solo escala los pesos, "
        "y como la frontera w·x = 0 no cambia al multiplicar w por una constante positiva, `a` "
        "**no afecta en absoluto** a la clasificacion resultante."
    )

    # ------------------------------------------------------------------
    # 2 y 3. Binaria frente a bipolar
    # ------------------------------------------------------------------
    reporte.seccion("2. Codificacion binaria frente a bipolar")
    filas = []
    for codificacion in ("binaria", "bipolar"):
        for nombre in ("AND", "OR"):
            conjunto = ds.compuerta(nombre, codificacion)
            objetivo = conjunto.d if codificacion == "bipolar" else 2 * conjunto.d - 1
            red = RedHebb(2, 1, razon_aprendizaje=1.0)
            red.entrenar(conjunto.X, objetivo)
            y = red.predecir(conjunto.X)
            filas.append({
                "codificacion": codificacion,
                "compuerta": nombre,
                "w0": float(red.W[0, 0]), "w1": float(red.W[0, 1]), "w2": float(red.W[0, 2]),
                "exactitud": mt.exactitud(y, objetivo),
                "patrones_mal": int(np.sum(y != objetivo)),
            })
            if codificacion == "bipolar" or nombre == "AND":
                fig, _ = vz.frontera_decision(
                    red.W[0], conjunto.X, objetivo,
                    titulo=f"Hebb sobre {nombre} ({codificacion}) --- exactitud "
                          f"{mt.exactitud(y, objetivo):.0%}",
                    etiquetas_clases=("d = +1", "d = −1"), margen=0.6,
                )
                ruta = vz.guardar(fig, FIGURAS / f"07_hebb_{nombre.lower()}_{codificacion}.png")
                reporte.figura(ruta, f"Frontera obtenida en una sola pasada sobre {nombre} "
                                     f"con entradas en codificacion {codificacion}.")

    df = pd.DataFrame(filas)
    print("\nRegla de Hebb sobre AND y OR:")
    print(df.to_string(index=False))
    reporte.tabla(df, "Resultado de la regla de Hebb segun la codificacion de las entradas",
                  nombre_csv="07_hebb_codificacion")

    # ------------------------------------------------------------------
    # 4. Por que falla en binario
    # ------------------------------------------------------------------
    reporte.seccion("3. Por que la codificacion decide el resultado")
    conjunto = ds.compuerta("AND", "binaria")
    objetivo = 2 * conjunto.d - 1
    red = RedHebb(2, 1)
    red.entrenar(conjunto.X, objetivo)
    detalle = pd.DataFrame({
        "patron": [f"({int(a)},{int(b)})" for a, b in conjunto.X],
        "d": objetivo,
        "aporte a w1 (d·x1)": objetivo * conjunto.X[:, 0],
        "aporte a w2 (d·x2)": objetivo * conjunto.X[:, 1],
        "aporte a w0 (d·1)": objetivo,
    })
    print("\nAportes de cada patron a los pesos (AND binaria):")
    print(detalle.to_string(index=False))
    reporte.tabla(detalle, "Aporte de cada patron a cada peso (AND en codificacion binaria)",
                  nombre_csv="07_aportes_binaria")
    reporte.texto(
        f"La suma de la columna `d·x1` es {detalle['aporte a w1 (d·x1)'].sum():.0f} y la de `d·x2` "
        f"es {detalle['aporte a w2 (d·x2)'].sum():.0f}: **ambos pesos quedan en cero**. La red "
        f"termina con w = {np.array2string(red.W[0], precision=0)}, que responde lo mismo ante "
        "cualquier entrada. El motivo es que el incremento es proporcional a x_i, de modo que "
        "todo patron con x_i = 0 es invisible para el peso w_i; y en el AND binario los unicos "
        "patrones con x_i = 1 se reparten entre las dos clases de forma que sus aportes se "
        "cancelan exactamente."
    )
    reporte.texto(
        "Con codificacion bipolar no hay entradas nulas: cada presentacion informa a **todos** los "
        "pesos, con signo. Por eso la clase insiste en trabajar con valores *\"bipolares o "
        "antisimetricos\"* al presentar la regla de Hebb. Es un detalle que parece cosmetico y "
        "decide entre funcionar y no funcionar."
    )

    # ------------------------------------------------------------------
    # 5. Memoria asociativa
    # ------------------------------------------------------------------
    reporte.seccion("4. Hebb como memoria asociativa: las letras X y O")
    letras = ds.letras_xo(con_ruido=0)
    red_letras = RedHebb(25, 2, razon_aprendizaje=1.0)
    red_letras.entrenar(letras.X, letras.d)
    exactitud = red_letras.exactitud(letras.X, letras.d)
    print(f"\nHebb sobre las letras X/O: exactitud sobre los prototipos = {exactitud:.0%}")

    fig, _ = vz.rejilla_retinas(
        [red_letras.W[0, 1:], red_letras.W[1, 1:]],
        ["Neurona X", "Neurona O"], n_columnas=2, divergente=True,
        sup_titulo="Pesos de Hebb: suma de correlaciones entrada x salida deseada",
    )
    ruta = vz.guardar(fig, FIGURAS / "07_hebb_letras_pesos.png")
    reporte.figura(ruta, "La regla de Hebb produce directamente la diferencia de plantillas "
                         "(X − O) para una neurona y (O − X) para la otra.")

    X_ref, O_ref = ds.letra("X"), ds.letra("O")
    df_letras = pd.DataFrame({
        "neurona": ["X", "O"],
        "w = (plantilla propia − plantilla rival)": [
            bool(np.array_equal(red_letras.W[0, 1:], X_ref - O_ref)),
            bool(np.array_equal(red_letras.W[1, 1:], O_ref - X_ref)),
        ],
        "w·x(X)": [float(red_letras.W[0, 1:] @ X_ref), float(red_letras.W[1, 1:] @ X_ref)],
        "w·x(O)": [float(red_letras.W[0, 1:] @ O_ref), float(red_letras.W[1, 1:] @ O_ref)],
    })
    reporte.tabla(df_letras, "Los pesos de Hebb son exactamente la diferencia de plantillas",
                  nombre_csv="07_hebb_letras")

    niveles = list(range(0, 13))
    rng = np.random.default_rng(7)
    exactitudes = []
    for k in niveles:
        aciertos = total = 0
        for _ in range(400):
            for indice, nombre in enumerate(("X", "O")):
                patron = ds.contaminar(ds.letra(nombre), k, semilla=int(rng.integers(1 << 30)))
                potencial = red_letras._con_sesgo(patron.reshape(1, -1)) @ red_letras.W.T
                aciertos += int(np.argmax(potencial) == indice)
                total += 1
        exactitudes.append(aciertos / total)

    fig, _ = vz.curva_aprendizaje(
        np.array(exactitudes),
        titulo="Memoria asociativa de Hebb: tolerancia al ruido",
        etiqueta_y="exactitud", etiqueta_x="pixeles invertidos (de 25)",
    )
    ruta = vz.guardar(fig, FIGURAS / "07_hebb_tolerancia.png")
    reporte.figura(ruta, "Exactitud al recuperar la letra correcta a partir de retinas degradadas "
                         "(400 ensayos por nivel y por letra).")
    reporte.tabla(pd.DataFrame({"pixeles_invertidos": niveles, "exactitud": exactitudes}),
                  "Tolerancia al ruido de la memoria hebbiana", nombre_csv="07_hebb_tolerancia")
    reporte.texto(
        f"Con una sola pasada y sin corregir un solo error, la red clasifica correctamente los dos "
        f"prototipos ({exactitud:.0%}) y mantiene una tolerancia al ruido comparable a la del "
        "perceptron del experimento 04. Esto es una **memoria asociativa** en el sentido de "
        "`claseRN01.md`: *\"una RNA que opere como memoria asociativa accede a la informacion por "
        "contenido, en tal sentido es capaz de recuperar informacion a partir de estimulos "
        "incompletos, ruidosos o parcialmente erroneos\"*. El precio de esa simplicidad aparece "
        "cuando los patrones almacenados no son ortogonales entre si: las correlaciones cruzadas "
        "se suman al peso y aparecen interferencias, algo que la regla perceptronica corrige por "
        "construccion y la de Hebb no."
    )

    # ------------------------------------------------------------------
    # 6. Comparacion con el perceptron
    # ------------------------------------------------------------------
    reporte.seccion("5. Comparacion con la regla perceptronica")
    filas = []
    for nombre in ("AND", "OR"):
        conjunto = ds.compuerta(nombre, "bipolar")
        hebb = RedHebb(2, 1).entrenar(conjunto.X, conjunto.d)
        perceptron = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=100)
        perceptron.entrenar(conjunto.X, conjunto.d)
        filas.append({
            "compuerta": nombre,
            "hebb: presentaciones": conjunto.n_patrones,
            "hebb: exactitud": hebb.exactitud(conjunto.X, conjunto.d),
            "hebb: margen": round(mt.margen_geometrico(hebb.W[0], conjunto.X, conjunto.d), 4),
            "perceptron: presentaciones": perceptron.epocas_usadas * conjunto.n_patrones,
            "perceptron: exactitud": mt.exactitud(perceptron.predecir(conjunto.X), conjunto.d),
            "perceptron: margen": round(mt.margen_geometrico(perceptron.W[0], conjunto.X, conjunto.d), 4),
        })
    df_comp = pd.DataFrame(filas)
    print("\nHebb frente a perceptron:")
    print(df_comp.to_string(index=False))
    reporte.tabla(df_comp, "Coste y calidad de ambas reglas sobre AND y OR (codificacion bipolar)",
                  nombre_csv="07_hebb_vs_perceptron")
    reporte.texto(
        "Sobre estos problemas la regla de Hebb obtiene el mismo resultado que la perceptronica "
        "con menos presentaciones, porque no necesita repasar el conjunto. La diferencia "
        "cualitativa es otra: **Hebb no garantiza nada**. Si el conjunto de entrenamiento no tiene "
        "la estructura adecuada --- como el AND en codificacion binaria --- la regla produce "
        "pesos inservibles y no hay forma de saberlo desde dentro del algoritmo, porque nunca "
        "comprueba sus propias respuestas. El perceptron, en cambio, tiene un teorema: si el "
        "problema es linealmente separable, converge."
    )

    # ------------------------------------------------------------------
    # 7. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("6. Conclusiones")
    reporte.lista([
        "La regla de Hebb aprende en **una sola pasada**, sin errores, iteraciones ni criterio de "
        "parada: es el aprendizaje mas barato que existe.",
        "Resuelve AND y OR en codificacion bipolar, pero **fracasa** en el AND binario, donde los "
        "aportes de los patrones se cancelan y todos los pesos quedan en cero.",
        "La razon de aprendizaje `a` es irrelevante: solo escala los pesos, sin mover la frontera.",
        "Como memoria asociativa funciona notablemente bien: los pesos son la diferencia de "
        "plantillas y recuperan la letra correcta a partir de retinas muy degradadas.",
        "Su debilidad es la ausencia de garantias: no verifica sus propias respuestas, de modo "
        "que puede fallar en silencio. Esa es la carencia que motiva la regla perceptronica.",
    ])

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
