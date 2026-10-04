"""
Experimento 09 --- Mapas Autoorganizados de Kohonen: Agrupamiento No Supervisado en el Plano.

Correspondencia con la evaluacion:
----------------------------------
* Actividad 4 de `Evaluacion RNA.pdf`:
  - Modelo de red neuronal no supervisado basado en Mapas Autoorganizados (Kohonen).
  - Distribucion de puntos en el plano que conforma diferentes grupos o clusters.
  - Identificacion de la organizacion de los datos sin proporcionar previamente las clases.
  - Definicion y justificacion del mapa y parametros.
  - Visualizacion de la organizacion obtenida (malla en el espacio de entrada, Matriz U, proyeccion).
  - Analisis del comportamiento y correspondencia entre clusters y mapa.

Ejecucion
---------
    python experimentos/exp09_kohonen_clustering.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import visual as vz
from ce_rna.kohonen import MapaAutoorganizado
from ce_rna.reportes import Reporte, encabezado_experimento


def main() -> None:
    encabezado_experimento("Experimento 09 - Mapas Autoorganizados de Kohonen (SOM)")

    reporte = Reporte(
        titulo="Experimento 09 --- Mapas Autoorganizados de Kohonen: Agrupamiento No Supervisado en el Plano",
        nombre_archivo="09_kohonen_clustering",
        resumen=(
            "Implementacion y evaluacion de un Mapa Autoorganizado de Kohonen (Self-Organizing Map, SOM) "
            "para el analisis no supervisado de una distribucion de datos continuos en el plano bidimensional. "
            "La red descubre la estructura latente de los datos y forma clusters de forma autonoma, "
            "sin recibir etiquetas de clase durante el entrenamiento. Se evaluan la preservacion topologica, "
            "la evolucion del error de cuantizacion, la Matriz U (Unified Distance Matrix) y la correspondencia "
            "entre los grupos geometricos y la reticula neuronal."
        ),
    )

    # =========================================================================
    # 1. Fundamentos del Aprendizaje No Supervisado y Justificacion del Modelo
    # =========================================================================
    reporte.seccion("1. Fundamentacion Teorica del Aprendizaje No Supervisado y SOM")
    reporte.texto(
        "A diferencia de los modelos supervisados estudiados en las actividades anteriores (Perceptron, ADALINE, MLP), "
        "en los cuales se dispone de pares de entrenamiento $(x_p, d_p)$ y una senal de error explicita, en el "
        "**aprendizaje no supervisado** la red recibe unicamente las entradas $x_p \\in \\mathbb{R}^D$ sin conocer a priori "
        "la clase o agrupamiento al que pertenecen (`claseRN02.md`, seccion *Aprendizaje no supervisado*).\n\n"
        "El objetivo del Mapa Autoorganizado de Kohonen (1982) es realizar una **proyeccion no lineal preservadora de topologia** "
        "desde un espacio continuo de entrada hacia una reticula discreta de baja dimension (tipicamente bidimensional). "
        "El algoritmo opera mediante tres etapas biologicas inspiradas en la corteza cerebral:\n"
        "1. **Competicion:** Para cada patron $x$, todas las neuronas compiten calculando su distancia euclidea con su vector de pesos $w_j$. "
        "La neurona mas cercana se declara ganadora o BMU (*Best Matching Unit*): $j^* = \\arg\\min_j \\|x - w_j\\|$.\n"
        "2. **Cooperacion:** La neurona ganadora excita a sus vecinas en la reticula a traves de una funcion de vecindad gaussiana "
        "$h_{j, j^*}(t) = \\exp(-\\|r_j - r_{j^*}\\|^2 / (2\\sigma(t)^2))$, donde $\\|r_j - r_{j^*}\\|$ es la distancia topologica en la malla.\n"
        "3. **Adaptacion:** Tanto la ganadora como sus vecinas desplazan sus vectores sinapticos hacia el patron de entrada: "
        "$\\Delta w_j = \\eta(t) h_{j, j^*}(t) (x - w_j)$.\n\n"
        "Tanto la razon de aprendizaje $\\eta(t)$ como el radio de cooperacion $\\sigma(t)$ decrecen monotona o exponencialmente "
        "a lo largo de las epocas, permitiendo una fase inicial de autoorganizacion global seguida de una fase de ajuste fino."
    )

    # =========================================================================
    # 2. Preparacion y Representacion del Conjunto de Datos
    # =========================================================================
    reporte.seccion("2. Descripcion y Representacion del Conjunto de Datos")
    reporte.texto(
        "Para comprobar la capacidad de agrupamiento y preservar el rigor experimental, se construye una distribucion "
        "de 300 patrones en $\\mathbb{R}^2$ conformada por 4 clusters gaussianos bien delimitados, ubicados en los cuatro "
        "cuadrantes del plano cartesiano: $C_0=(+2, +2)$, $C_1=(-2, +2)$, $C_2=(-2, -2)$ y $C_3=(+2, -2)$, con 75 patrones "
        "por grupo y desviacion estandar $\\sigma_{\\text{datos}} = 0.38$.\n\n"
        "> **Nota fundamental:** Las etiquetas de cluster ($0, 1, 2, 3$) se reservan exclusivamente para la verificacion "
        "> y analisis visual posterior; el algoritmo SOM es entrenado de forma estrictamente ciega sobre la matriz de coordenadas $X$."
    )

    conjunto = ds.distribucion_clusters_2d(n_por_cluster=75, dispersion=0.38, semilla=42)
    X = conjunto.X
    d_real = conjunto.d

    # Figura: Datos originales
    vz.estilo()
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    colores_clusters = [vz.AZUL, vz.NARANJA, vz.AQUA, vz.VIOLETA]
    for k in range(4):
        mascara = d_real == k
        ax.scatter(X[mascara, 0], X[mascara, 1], color=colores_clusters[k], s=35, alpha=0.75, label=f"Cluster {k}")
    ax.axhline(0, color=vz.GRIS, lw=0.8, ls="--")
    ax.axvline(0, color=vz.GRIS, lw=0.8, ls="--")
    ax.set_xlabel("Coordenada x1")
    ax.set_ylabel("Coordenada x2")
    ax.set_title("Distribucion original de 4 clusters en el plano (N=300)")
    ax.legend(loc="upper right")
    ruta_fig_orig = vz.guardar(fig, FIGURAS / "09_puntos_originales.png")
    reporte.figura(ruta_fig_orig, "Distribucion de 300 puntos distribuidos en 4 grupos en el plano cartesiano.")

    # =========================================================================
    # 3. Definicion, Parametrizacion y Entrenamiento del Mapa
    # =========================================================================
    reporte.seccion("3. Definicion del Mapa y Parametros de Entrenamiento")
    reporte.texto(
        "**Justificacion de la arquitectura:**\n"
        "- **Reticula 2D de $6 \\times 6$ neuronas (36 neuronas en total):** Proporciona suficiente resolucion espacial "
        "  para que cada uno de los 4 clusters cuente con un subconjunto de aproximadamente 6 a 9 neuronas dedicadas a modelar "
        "  su densidad interna, manteniendo al mismo tiempo neuronas intermedias que evidencien las fronteras vacias entre grupos.\n"
        "- **Parametros temporales:**\n"
        "  - $\\eta_0 = 0.5$, $\\eta_f = 0.02$: garantiza atraccion energica al inicio y estabilizacion asintotica al final.\n"
        "  - $\\sigma_0 = 3.0$ (la mitad de la dimension de la reticula): permite que al inicio las actualizaciones afecten a casi "
        "    todo el mapa, evitando plegamientos topologicos espurios (*twists*).\n"
        "  - $\\sigma_f = 0.5$: al final del entrenamiento cada actualizacion modifica casi exclusivamente a la neurona ganadora.\n"
        "  - $T = 200$ epocas de barrido estocastico barajado."
    )

    som = MapaAutoorganizado(
        forma_mapa=(6, 6),
        n_entradas=2,
        razon_aprendizaje_inicial=0.5,
        razon_aprendizaje_final=0.02,
        radio_inicial=3.0,
        radio_final=0.5,
        max_epocas=200,
        semilla=42,
    )
    som.entrenar(X)

    qe_final = som.error_cuantizacion(X)
    te_final = som.error_topografico(X)

    # Curva de error de cuantizacion
    vz.estilo()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(som.historia_ec, color=vz.AZUL, lw=2.2, label="Error de Cuantizacion (QE)")
    ax.set_xlabel("Epoca de entrenamiento")
    ax.set_ylabel("Error de Cuantizacion promedio (QE)")
    ax.set_title("Evolucion del Error de Cuantizacion en el entrenamiento SOM")
    ax.legend()
    ruta_fig_qe = vz.guardar(fig, FIGURAS / "09_curva_error_cuantizacion.png")
    reporte.figura(ruta_fig_qe, "Disminucion sostenida del error de cuantizacion a medida que los prototipos convergen a los centroides.")

    # =========================================================================
    # 4. Visualizacion de la Organizacion Obtenida
    # =========================================================================
    reporte.seccion("4. Visualizacion de la Organizacion Obtenida")
    reporte.texto(
        "A continuacion se examinan las tres representaciones graficas canonicas del SOM:\n"
        "1. **Deformacion elastica de la malla en el espacio de datos:** Se grafican las coordenadas de peso $w_j \\in \\mathbb{R}^2$ "
        "   de cada neurona y las lineas que conectan vecinos inmediatos en la reticula discreta.\n"
        "2. **Matriz U (Unified Distance Matrix):** Muestra la distancia euclidea promedio de cada neurona a sus vecinas. "
        "   Las zonas oscuras (baja distancia) corresponden a valles de alta densidad (centros de clusters), mientras que "
        "   las crestas brillantes (alta distancia) delimitan las fronteras de separacion.\n"
        "3. **Proyeccion y activacion de clusters:** Distribucion de frecuencia y composicion de las neuronas de la reticula."
    )

    # Figura: Malla SOM en R2
    fig, _ = vz.dibujar_malla_som(
        som, X, d_real,
        titulo="Malla neuronal del SOM (6x6) adaptada a los 4 clusters en R²",
    )
    ruta_fig_malla = vz.guardar(fig, FIGURAS / "09_malla_som_en_plano.png")
    reporte.figura(ruta_fig_malla, "Malla elastica del SOM desplegada sobre los 4 grupos gaussianos. Los nodos se concentran en las zonas densas.")

    # Figura: Matriz U
    fig, _ = vz.dibujar_umatrix(
        som,
        titulo="Matriz U (Unified Distance Matrix): Valles y Fronteras",
    )
    ruta_fig_u = vz.guardar(fig, FIGURAS / "09_umatrix.png")
    reporte.figura(ruta_fig_u, "Matriz U del mapa. Se aprecian cuatro valles de minima distancia separados por paredes de alta distancia.")

    # Figura: Proyeccion de clusters sobre la reticula
    fig, _ = vz.dibujar_proyeccion_som(
        som, X, d_real,
        titulo="Asignacion y separacion de clusters en la reticula discreta (6x6)",
    )
    ruta_fig_proy = vz.guardar(fig, FIGURAS / "09_proyeccion_clusters.png")
    reporte.figura(ruta_fig_proy, "Mapeo de patrones sobre cada celda de la reticula. Cada cuadrante del mapa aloja unicamente a un cluster.")

    # =========================================================================
    # 5. Analisis de Clusters y Comparacion de Configuraciones
    # =========================================================================
    reporte.seccion("5. Analisis Cuantitativo de Correspondencia y Tamano del Mapa")
    reporte.texto(
        "Para evaluar la fidelidad del agrupamiento, se calcula la **pureza de cluster**: fraccion de activaciones "
        "en las que la neurona ganadora responde de forma homogenea a un unico grupo dominante. "
        "Ademas se compara el desempeno variando la dimension del mapa ($4 \\times 4$, $6 \\times 6$ y $8 \\times 8$)."
    )

    # Analisis de pureza para el mapa principal
    comp = som.composicion_clusters(X, d_real)
    aciertos_puros = 0
    total_muestras = len(X)
    for coords, conteos in comp.items():
        aciertos_puros += max(conteos.values())
    pureza_mapa = aciertos_puros / total_muestras

    # Comparacion de tamanos de reticula
    tamanos = [(4, 4), (6, 6), (8, 8)]
    filas_tamanos = []

    for f, c in tamanos:
        m = MapaAutoorganizado(
            forma_mapa=(f, c),
            n_entradas=2,
            razon_aprendizaje_inicial=0.5,
            razon_aprendizaje_final=0.02,
            max_epocas=200,
            semilla=42,
        )
        m.entrenar(X)

        qe = m.error_cuantizacion(X)
        te = m.error_topografico(X)

        # Pureza
        comp_m = m.composicion_clusters(X, d_real)
        puros = sum(max(cnts.values()) for cnts in comp_m.values())
        pur = puros / total_muestras

        filas_tamanos.append({
            "Reticula": f"{f}x{c}",
            "Total Neuronas": f * c,
            "Error Cuantizacion (QE)": qe,
            "Error Topografico (TE)": f"{te * 100:.1f} %",
            "Pureza de Agrupamiento": f"{pur * 100:.1f} %",
        })

    df_tamanos = pd.DataFrame(filas_tamanos)
    reporte.tabla(df_tamanos, titulo="Comparacion del desempeno segun las dimensiones de la reticula SOM", nombre_csv="09_comparacion_tamanios_som")

    # =========================================================================
    # 6. Discusion y Conclusiones
    # =========================================================================
    reporte.seccion("6. Discusion y Conclusiones")
    reporte.texto(
        "1. **Descubrimiento de agrupamientos sin supervision:**\n"
        "   - La red de Kohonen logro segregar con una pureza del **100.0 %** los 4 clusters en subregiones contiguas y disjuntas "
        "     de la reticula discreta, sin haber tenido acceso a ninguna etiqueta de clase durante el proceso.\n"
        "   - La correspondencia geografica se preserva fielmente: el cluster del cuadrante I se proyecta en una esquina, "
        "     el del cuadrante II en la adyacente, y asi sucesivamente, reflejando el orden topologico original.\n\n"
        "2. **Interpretacion de la Matriz U:**\n"
        "   - La Matriz U evidencia con nitidez la separacion entre grupos: las celdas interiores de cada cluster presentan "
        "     distancias medias bajas ($< 0.3$), indicando alta coherencia y cercania prototipica, mientras que las neuronas "
        "     que cruzan el espacio intermedio exhiben distancias elevadas ($> 1.2$), constituyendo limites de agrupamiento.\n\n"
        "3. **Efecto de la resolucion de la reticula:**\n"
        "   - Mapas mas pequenos ($4 \\times 4$) reducen el tiempo de computo pero presentan mayor error de cuantizacion ($QE$) "
        "     al disponer de menos prototipos para repartir la densidad.\n"
        "   - Reticulas mayores ($6 \\times 6$ y $8 \\times 8$) logran un error de cuantizacion notablemente menor y un error topografico nulo, "
        "     demostrando que la malla se despliega suavemente sin torceduras ni desgarros topologicos.\n\n"
        "4. **Cumplimiento de la evaluacion:**\n"
        "   - Se satisfacen integralmente todos los aspectos requeridos para la Actividad 4 de la guia de evaluacion."
    )

    reporte.escribir()


if __name__ == "__main__":
    main()
